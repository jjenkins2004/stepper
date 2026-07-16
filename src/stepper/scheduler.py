"""Dependency scheduler: run a stage's steps concurrently, in dependency order.

A stage's `steps = (...)` tuple is a *membership* set, not a run order. `Scheduler`
reads each step's `depends()` markers into an in-stage dependency DAG, validates it at
construction, then `run(runners)` executes it: launch every step whose upstreams have
finished, wait for the next completion, launch whatever that unblocks — so independent
steps run at the same time (max parallelism). A failed step never enters `done`, so its
dependents never become runnable (they skip) while independent branches finish.

The scheduler owns the *ordering* and the *loop*; it does not know how a step runs. The
caller passes `run_step` — a callback that runs one step by name and returns its result —
where fetch/run/persist lives. Cross-stage deps (owned by another stage) are disk inputs,
not scheduling edges, so they're excluded from the graph.

Validation happens when the scheduler is built (not during a run):
- every `depends()` target resolves to a real step (this stage or another stage);
- the in-stage graph has no cycles.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Collection, Coroutine, Mapping
from typing import Any

from stepper.step import Step

_LOGGER = logging.getLogger(__name__)

# Runs one step by name and returns its result; the scheduler decides *when*.
RunStep = Callable[[str], Coroutine[Any, Any, Any]]


class Scheduler:
    """Runs one stage's steps by their dependency DAG. Built from the `steps` tuple
    (which fixes the graph); `run(run_step)` supplies how to execute a step by name."""

    def __init__(self, steps: tuple[Step[Any], ...], *, label: str = "") -> None:
        """Build the dependency DAG from `steps` (which fixes run order) and validate it
        now — unknown dep target or cycle raises here, before any run. `label` tags error
        and warning messages, usually the stage class name."""
        self._label = label
        self._names: list[str] = [s.name for s in steps]
        self._upstreams: dict[str, set[str]] = _intra_stage_upstreams(steps)
        self._validate(steps)

    async def run(self, run_step: RunStep, *, fail_fast: bool = False) -> list[Any]:
        """Run every step, respecting deps, with maximum parallelism. `run_step(name)`
        runs one step by name and returns its result — the scheduler decides *when* each
        name runs. Returns completed results in declaration order. `fail_fast=True`
        cancels in-flight steps and re-raises on the first failure; the default records
        the failure, skips its dependents, and lets independent branches finish."""
        done: dict[str, Any] = {}                             # completed OK -> result
        failed: set[str] = set()                             # raised
        started: set[str] = set()                            # launched (running, done, or failed)
        running: dict[asyncio.Task[Any], str] = {}           # in-flight task -> step name

        while True:
            for name in self._unblocked(done, started):
                started.add(name)
                running[asyncio.create_task(run_step(name))] = name

            if not running:
                break  # nothing runnable and nothing in flight — the rest is blocked by a failure

            completed, _ = await asyncio.wait(running.keys(), return_when=asyncio.FIRST_COMPLETED)
            for task in completed:
                name = running.pop(task)
                exc = task.exception()
                if exc is None:
                    done[name] = task.result()
                else:
                    failed.add(name)          # dependents stay blocked -> skipped
                    if fail_fast:
                        await _cancel(running)
                        raise exc

        self._warn_incomplete(failed, started)
        # Declaration order for readable output; execution order came from the deps.
        return [done[name] for name in self._names if name in done]

    def _unblocked(self, done: Collection[str], started: Collection[str]) -> list[str]:
        """Steps whose in-stage upstreams are all in `done` and that haven't started."""
        return [
            name for name in self._names
            if name not in started and self._upstreams[name].issubset(done)
        ]

    def _warn_incomplete(self, failed: set[str], started: Collection[str]) -> None:
        skipped = set(self._names) - set(started)
        if failed or skipped:
            _LOGGER.warning(
                "%s: %d step(s) failed (%s); %d dependent(s) skipped (%s).",
                self._label or "scheduler",
                len(failed), ", ".join(sorted(failed)) or "-",
                len(skipped), ", ".join(sorted(skipped)) or "-",
            )

    def _validate(self, steps: tuple[Step[Any], ...]) -> None:
        members = set(steps)
        where = f"{self._label} " if self._label else ""

        # (1) Every dep resolves: a step here, or one owned by another stage (a
        # cross-stage input). An unclaimed target is a typo / forgotten `steps` entry.
        for s in steps:
            for dep in s.dependencies().values():
                if dep not in members and dep.owner is None:
                    raise TypeError(
                        f"{where}step {s.name!r} depends on step {dep.name!r}, "
                        f"which is not in this stage or any other stage."
                    )

        # (2) No cycles: DFS over in-stage edges; a back-edge to a step still on the
        # current path is a cycle — raise naming the loop.
        WHITE, GRAY, BLACK = 0, 1, 2
        color = dict.fromkeys(self._upstreams, WHITE)
        path: list[str] = []

        def visit(name: str) -> None:
            color[name] = GRAY
            path.append(name)
            for up in self._upstreams[name]:
                if color[up] == GRAY:
                    cycle = path[path.index(up):] + [up]
                    raise TypeError(f"{where}dependency cycle: {' -> '.join(cycle)}")
                if color[up] == WHITE:
                    visit(up)
            path.pop()
            color[name] = BLACK

        for name in self._upstreams:
            if color[name] == WHITE:
                visit(name)


async def _cancel(running: Mapping[asyncio.Task[Any], str]) -> None:
    """Cancel in-flight tasks and wait for them to settle (used by fail_fast)."""
    for task in running:
        task.cancel()
    await asyncio.gather(*running, return_exceptions=True)


def _intra_stage_upstreams(steps: tuple[Step[Any], ...]) -> dict[str, set[str]]:
    """Map each step to the names of its upstream steps *in this tuple* — the
    scheduling edges. Cross-stage deps (owned by another stage) are excluded."""
    members = set(steps)
    return {
        s.name: {dep.name for dep in s.dependencies().values() if dep in members}
        for s in steps
    }
