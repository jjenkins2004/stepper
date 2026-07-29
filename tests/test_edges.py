"""Edge-driven stages, end to end: control flow, back-edge deps, resume, entry points.

Graph *shape* validation lives in test_graph_validation.py; cursor failure modes and
backend/hook/Pipeline interactions live in test_graph_runtime.py.

No ``from __future__ import annotations`` here for the same reason as ``_helpers`` —
these stages are declared inside test functions, so stringized annotations would have
no module namespace to resolve against.

Note what these stages *don't* have: any framework-owned notion of a round. A loop that
needs to count counts in its own model — ``Count.n`` is a field a step wrote, read back
on the next pass through the graph's back edge.
"""

import pytest
from pydantic import BaseModel

from stepper import (
    EXIT,
    START,
    DiskPersistService,
    Stage,
    depends,
    edge,
    optional_depends,
    step,
)
from stepper.stage import _Cursor


class Count(BaseModel):
    n: int


def test_loop_counts_in_its_own_model(run, tmp_path):
    """head -> tail -> head. `head` reads the previous pass's `tail` by name (the back
    edge: `tail` is defined below it), None on the first pass, so the count belongs
    entirely to the steps."""
    seen = []

    class CountStage(Stage):
        @step
        async def head(self, prev=optional_depends("tail")) -> Count:
            n = 0 if prev is None else prev.n + 1
            seen.append(n)
            return Count(n=n)

        @step
        async def tail(self, c=depends(head)) -> Count:
            return Count(n=c.n)

        steps = (head, tail)
        edges = (
            edge(START).to(head),
            edge(head).to(tail),
            edge(tail).when(lambda r: r.n >= 2).to(EXIT).otherwise(head),
        )

    stage = CountStage(persist_service=DiskPersistService(base_dir=tmp_path))
    results = run(stage.run_steps())

    assert seen == [0, 1, 2]                   # three passes, counted by the steps
    assert [r.n for r in results] == [2, 2]    # last value of head, then of tail


def test_straight_line_graph_with_a_loop_in_the_middle(run, tmp_path):
    """An edge-driven stage declares *everything*, so setup and report are edges too —
    there is no DAG left over to infer."""
    order = []

    class MixedStage(Stage):
        @step
        async def setup(self) -> Count:
            order.append("setup")
            return Count(n=10)

        @step
        async def body(self, base=depends(setup), prev=optional_depends("echo")) -> Count:
            n = base.n if prev is None else prev.n + 1
            order.append(f"body{n}")
            return Count(n=n)

        @step
        async def echo(self, c=depends(body)) -> Count:
            return Count(n=c.n)

        @step
        async def report(self, c=depends(body)) -> str:
            order.append("report")
            return f"final {c.n}"

        steps = (setup, body, echo, report)
        edges = (
            edge(START).to(setup),
            edge(setup).to(body),
            edge(body).to(echo),
            edge(echo).when(lambda r: r.n >= 12).to(report).otherwise(body),
            edge(report).to(EXIT),
        )

    stage = MixedStage(persist_service=DiskPersistService(base_dir=tmp_path))
    run(stage.run_steps())

    assert order == ["setup", "body10", "body11", "body12", "report"]
    assert (tmp_path / "Mixed" / "report.txt").read_text() == "final 12"


def test_branch_order_first_match_wins(run, tmp_path):
    ran = []

    class BranchStage(Stage):
        @step
        async def start(self) -> Count:
            ran.append("start")
            return Count(n=0)

        @step
        async def left(self, c=depends(start)) -> str:
            ran.append("left")
            return "left"

        @step
        async def right(self, c=depends(start)) -> str:
            ran.append("right")
            return "right"

        steps = (start, left, right)
        edges = (
            edge(START).to(start),
            # Both predicates match; the one declared first must win.
            edge(start).when(lambda r: True).to(left).when(lambda r: True).to(right).otherwise(right),
            edge(left).to(EXIT),
            edge(right).to(EXIT),
        )

    stage = BranchStage(persist_service=DiskPersistService(base_dir=tmp_path))
    results = run(stage.run_steps())
    assert ran == ["start", "left"]                # `right` never executed
    assert results == [Count(n=0), "left"]         # only the steps this run touched
    assert not (tmp_path / "Branch" / "right.txt").exists()


def test_resume_restarts_at_the_failed_step(run, tmp_path):
    """A crash mid-graph leaves the cursor on the failed step; re-running the stage picks
    up there rather than at START, and the counters carry over because they live in the
    steps' own persisted output."""
    calls = []
    boom = {"armed": True}

    class FlakyStage(Stage):
        @step
        async def first(self, prev=optional_depends("second")) -> Count:
            n = 0 if prev is None else prev.n + 1
            calls.append(("first", n))
            return Count(n=n)

        @step
        async def second(self, c=depends(first)) -> Count:
            calls.append(("second", c.n))
            if c.n == 1 and boom["armed"]:
                boom["armed"] = False
                raise RuntimeError("crash")
            return Count(n=c.n)

        steps = (first, second)
        edges = (
            edge(START).to(first),
            edge(first).to(second),
            edge(second).when(lambda r: r.n >= 2).to(EXIT).otherwise(first),
        )

    persist = DiskPersistService(base_dir=tmp_path)

    with pytest.raises(RuntimeError, match="crash"):
        run(FlakyStage(persist_service=persist).run_steps())
    assert calls == [("first", 0), ("second", 0), ("first", 1), ("second", 1)]

    calls.clear()
    run(FlakyStage(persist_service=persist).run_steps())

    # Restarts at `second`, not back at `first` and not back at count 0.
    assert calls == [("second", 1), ("first", 2), ("second", 2)]


def test_completed_graph_does_not_resume(run, tmp_path):
    """A finished graph marks its cursor done, so the next run starts from START."""
    calls = []

    class OnceStage(Stage):
        @step
        async def only(self, prev=optional_depends("mirror")) -> Count:
            n = 0 if prev is None else prev.n + 1
            calls.append(n)
            return Count(n=n)

        @step
        async def mirror(self, c=depends(only)) -> Count:
            return Count(n=c.n)

        steps = (only, mirror)
        edges = (
            edge(START).to(only),
            edge(only).to(mirror),
            edge(mirror).when(lambda r: r.n >= 1).to(EXIT).otherwise(only),
        )

    persist = DiskPersistService(base_dir=tmp_path)
    run(OnceStage(persist_service=persist).run_steps())
    assert calls == [0, 1]

    calls.clear()
    run(OnceStage(persist_service=persist).run_steps())
    assert calls == [2]         # START again, carrying on from what's persisted


def test_single_step_run_does_not_advance_the_cursor(run, tmp_path):
    class ManualStage(Stage):
        @step
        async def a(self) -> Count:
            return Count(n=1)

        @step
        async def b(self, c=depends(a)) -> Count:
            return Count(n=c.n + 1)

        steps = (a, b)
        edges = (
            edge(START).to(a),
            edge(a).to(b),
            edge(b).when(lambda r: r.n >= 2).to(EXIT).otherwise(a),
        )

    stage = ManualStage(persist_service=DiskPersistService(base_dir=tmp_path))

    # Running one step is well-defined — there's only ever one value per step, so "which
    # pass" never comes up. It just isn't a graph advance.
    run(stage.run_step("a"))
    assert run(stage.run_step("b")).n == 2
    assert not (tmp_path / "Manual" / "_loop_cursor.json").exists()


def test_max_steps_fuse_raises(run, tmp_path):
    class SpinStage(Stage):
        @step
        async def spin(self) -> Count:
            return Count(n=0)

        steps = (spin,)
        edges = (
            edge(START).to(spin),
            edge(spin).when(lambda r: False).to(EXIT).otherwise(spin),
        )
        max_steps = 3

    stage = SpinStage(persist_service=DiskPersistService(base_dir=tmp_path))
    with pytest.raises(RuntimeError, match="ran 3 steps without reaching EXIT"):
        run(stage.run_steps())


def test_hooks_fire_once_per_step_execution(run, tmp_path):
    """A step on a cycle runs many times, so its hook runs many times — nothing about the
    Hooks protocol changed."""
    from contextlib import contextmanager

    fired = []

    class CountingHook:
        @contextmanager
        def step(self, *, stage_name, step_name, input_type, output_type):
            fired.append(step_name)
            yield None

        @contextmanager
        def stage(self, *, stage_name, step_count):
            yield

    class HookedStage(Stage):
        @step
        async def tick(self, prev=optional_depends("tock")) -> Count:
            return Count(n=0 if prev is None else prev.n + 1)

        @step
        async def tock(self, c=depends(tick)) -> Count:
            return Count(n=c.n)

        steps = (tick, tock)
        edges = (
            edge(START).to(tick),
            edge(tick).to(tock),
            edge(tock).when(lambda r: r.n >= 2).to(EXIT).otherwise(tick),
        )

    stage = HookedStage(
        persist_service=DiskPersistService(base_dir=tmp_path), hooks=CountingHook()
    )
    run(stage.run_steps())
    assert fired == ["tick", "tock", "tick", "tock", "tick", "tock"]


def test_no_edges_still_runs_as_a_concurrent_dag(run, tmp_path):
    """The other model, untouched: no `edges`, so `depends()` is the order."""
    from _helpers import AStage, BStage

    assert BStage._graph is None and BStage._scheduler is not None
    persist = DiskPersistService(base_dir=tmp_path)
    run(AStage(persist_service=persist).run_steps())
    results = run(BStage(persist_service=persist).run_steps())
    assert results[1] == "got 2"
    assert not (tmp_path / "B" / "_loop_cursor.json").exists()


def _abc_stage():
    """a -> b -> c -> (EXIT when n >= 3, else a). Every step bumps the count it was
    handed, so where you enter the graph is visible in what runs."""
    seen = []

    class AbcStage(Stage):
        @step
        async def a(self, prev=optional_depends("c")) -> Count:
            n = 0 if prev is None else prev.n + 1
            seen.append(("a", n))
            return Count(n=n)

        @step
        async def b(self, c_=depends(a)) -> Count:
            seen.append(("b", c_.n))
            return Count(n=c_.n)

        @step
        async def c(self, c_=depends(b)) -> Count:
            seen.append(("c", c_.n))
            return Count(n=c_.n)

        steps = (a, b, c)
        edges = (
            edge(START).to(a),
            edge(a).to(b),
            edge(b).to(c),
            edge(c).when(lambda r: r.n >= 2).to(EXIT).otherwise(a),
        )

    return AbcStage, seen


def test_start_at_enters_the_graph_and_follows_edges(run, tmp_path):
    AbcStage, seen = _abc_stage()
    persist = DiskPersistService(base_dir=tmp_path)

    run(AbcStage(persist_service=persist).run_steps())
    assert seen == [("a", 0), ("b", 0), ("c", 0), ("a", 1), ("b", 1), ("c", 1),
                    ("a", 2), ("b", 2), ("c", 2)]

    seen.clear()
    # Enter at `b` against what's persisted (a=2), then keep following edges to EXIT.
    run(AbcStage(persist_service=persist).run_steps(start_at="b"))
    assert seen == [("b", 2), ("c", 2)]


def test_start_at_overrides_a_saved_cursor(run, tmp_path):
    AbcStage, seen = _abc_stage()
    persist = DiskPersistService(base_dir=tmp_path)
    run(AbcStage(persist_service=persist).run_steps())

    stage = AbcStage(persist_service=persist)
    stage._save_cursor(_Cursor(next="a"))       # a crash would have left this
    seen.clear()
    run(stage.run_steps(start_at="c"))          # explicit entry wins
    assert seen == [("c", 2)]


def test_start_at_rejects_a_dag_stage_and_an_unknown_step(run, tmp_path):
    from _helpers import BStage

    AbcStage, _ = _abc_stage()
    persist = DiskPersistService(base_dir=tmp_path)

    with pytest.raises(ValueError, match="start_at needs a stage that declares edges"):
        run(BStage(persist_service=persist).run_steps(start_at="note"))

    with pytest.raises(ValueError, match="unknown step 'nope'"):
        run(AbcStage(persist_service=persist).run_steps(start_at="nope"))


def test_pipeline_follow_edges(run, tmp_path):
    from stepper import Pipeline

    AbcStage, seen = _abc_stage()
    p = Pipeline(name="p", output_root=tmp_path, stages={"abc": lambda ps: AbcStage(persist_service=ps)})

    run(p.run(stage="abc"))
    seen.clear()

    assert run(p.run(stage="abc", step="b")).n == 2        # just that step, its value back
    assert seen == [("b", 2)]

    seen.clear()
    out = run(p.run(stage="abc", step="b", follow_edges=True))   # enter there, keep going
    assert seen == [("b", 2), ("c", 2)]
    assert [v.n for v in out] == [2, 2]                   # the steps this run executed

    with pytest.raises(ValueError, match="follow_edges needs a step"):
        run(p.run(stage="abc", follow_edges=True))
