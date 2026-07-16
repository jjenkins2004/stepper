"""Scheduler — dependency DAG build/validation + the concurrent run loop.

Unit tests drive `SomeStage._scheduler.run(callback)` directly with a recording/gate
callback — no persist, no Stage instance (`_scheduler` is a ClassVar). Concurrency is
proven deterministically with asyncio barriers/gates (serialized execution => deadlock
=> `wait_for` timeout => failure), never with timing sleeps.

No `from __future__ import annotations` — `Step` reads raw `fn.__annotations__`.
"""

import asyncio

import pytest

from stepper.scheduler import Scheduler
from stepper.stage import Stage
from stepper.step import Step, depends, step

from _helpers import AStage, Item


# --- graph build / validation (at Scheduler construction) ---

def test_upstreams_exclude_cross_stage_dep_keep_same_stage():
    class Mix(Stage):
        @step
        async def base(self) -> Item:
            return Item(value=0)

        @step
        async def mixed(self, x=depends(AStage.prod), y=depends(base)) -> Item:
            return y

        steps = (base, mixed)

    # cross-stage dep (AStage.prod) is a disk input, not a scheduling edge; same-stage dep is.
    assert Mix._scheduler._upstreams == {"base": set(), "mixed": {"base"}}


def test_missing_dependency_raises():
    @step
    async def orphan(self) -> Item:  # never listed in any stage's steps -> unclaimed
        return Item(value=0)

    with pytest.raises(TypeError, match="not in this stage"):

        class Broken(Stage):
            @step
            async def main(self, x=depends(orphan)) -> Item:
                return Item(value=1)

            steps = (main,)


def test_cycle_detected_and_named():
    # depends() can't express a cycle (forward-reference), so hand-wire two steps that
    # point at each other via __defaults__ and build the Scheduler directly.
    async def af(self, up=None): ...
    async def bf(self, up=None): ...

    sa, sb = Step(af), Step(bf)
    af.__defaults__ = (sb,)  # af depends on bf
    bf.__defaults__ = (sa,)  # bf depends on af

    with pytest.raises(TypeError, match="dependency cycle"):
        Scheduler((sa, sb), label="Cyc")


# --- the run loop (callback-driven, no persist) ---

def test_execution_follows_deps_return_follows_declaration_order(run):
    order: list[str] = []

    class Chain(Stage):
        @step
        async def a(self) -> Item:
            return Item(value=1)

        @step
        async def b(self, up=depends(a)) -> Item:
            return Item(value=2)

        @step
        async def c(self, up=depends(b)) -> Item:
            return Item(value=3)

        steps = (c, a, b)  # scrambled: _names = [c, a, b], run order = [a, b, c]

    async def cb(name: str) -> str:
        order.append(name)
        return name

    result = run(Chain._scheduler.run(cb))
    assert order == ["a", "b", "c"]    # execution follows deps, tuple order ignored
    assert result == ["c", "a", "b"]   # return follows declaration order, NOT completion order


def test_diamond_runs_upstreams_concurrently_and_joins(run):
    order: list[str] = []
    barrier = asyncio.Barrier(2)

    class Diamond(Stage):
        @step
        async def a(self) -> Item:
            return Item(value=0)

        @step
        async def b(self, up=depends(a)) -> Item:
            return Item(value=1)

        @step
        async def c(self, up=depends(a)) -> Item:
            return Item(value=2)

        @step
        async def d(self, x=depends(b), y=depends(c)) -> Item:
            return Item(value=3)

        steps = (d, c, b, a)  # scrambled

    async def cb(name: str) -> str:
        if name in ("b", "c"):
            await barrier.wait()  # both must arrive -> concurrency (serialized => timeout)
        order.append(name)
        return name

    run(asyncio.wait_for(Diamond._scheduler.run(cb), 1.0))
    assert order[0] == "a"            # root first
    assert order[-1] == "d"           # join waits for both b and c
    assert set(order) == {"a", "b", "c", "d"}


def test_failed_step_skips_join_and_dependents_independent_completes(run):
    order: list[str] = []

    class F(Stage):
        @step
        async def a(self) -> Item:
            return Item(value=0)

        @step
        async def b(self, up=depends(a)) -> Item:
            return Item(value=1)

        @step
        async def boom(self, up=depends(a)) -> Item:  # fails in the callback
            return Item(value=9)

        @step
        async def join(self, x=depends(b), y=depends(boom)) -> Item:  # needs BOTH -> skipped
            return Item(value=2)

        @step
        async def indep(self) -> Item:
            return Item(value=7)

        steps = (a, b, boom, join, indep)

    async def cb(name: str) -> str:
        order.append(name)
        if name == "boom":
            raise RuntimeError("boom failed")
        return name

    result = run(asyncio.wait_for(F._scheduler.run(cb), 1.0))
    # boom fails; join needs ALL upstreams (b AND boom) so it skips; b and indep still complete.
    assert result == ["a", "b", "indep"]  # declaration order, boom + join absent
    assert "join" not in order            # never launched (a failed upstream, not just any)


def test_fail_fast_raises_and_cancels_inflight(run):
    slow_started = asyncio.Event()
    saw_cancel = {"v": False}

    class FF(Stage):
        @step
        async def slow(self) -> Item:
            return Item(value=1)

        @step
        async def boom(self) -> Item:
            return Item(value=2)

        steps = (slow, boom)

    async def cb(name: str) -> str:
        if name == "slow":
            slow_started.set()
            try:
                await asyncio.Event().wait()  # block until cancelled
            except asyncio.CancelledError:
                saw_cancel["v"] = True
                raise
        else:  # boom fails only after slow is in-flight (gate removes the race)
            await slow_started.wait()
            raise RuntimeError("boom failed")
        return name

    with pytest.raises(RuntimeError, match="boom failed"):
        run(asyncio.wait_for(FF._scheduler.run(cb, fail_fast=True), 1.0))
    assert saw_cancel["v"] is True  # the running sibling was cancelled, not left leaking
