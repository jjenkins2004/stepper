"""Edge-driven stages at run time: the cursor's failure modes, the fuse boundary, and
the interactions (hooks, Pipeline, in-memory backend, mixed stages) that only show up
once something actually runs.

Shape-level validation lives in test_graph_validation.py; the happy paths live in
test_edges.py. This file is the awkward middle: what happens when the checkpoint is
missing, stale, corrupt, or unwritable, and when a run enters somewhere unusual.

No ``from __future__ import annotations`` — these stages are declared inside functions.
"""

import pytest
from pydantic import BaseModel

from stepper import (
    EXIT,
    START,
    DiskPersistService,
    InMemoryPersistService,
    Pipeline,
    Stage,
    StepReport,
    depends,
    edge,
    optional_depends,
    step,
)
from stepper.stage import _CURSOR_KEY, _Cursor


class V(BaseModel):
    n: int


def _chain(calls):
    """s1 -> s2 -> s3 -> (EXIT when n >= 2, else s1), counting in the model."""

    class ChainStage(Stage):
        @step
        async def s1(self, prev=optional_depends("s3")) -> V:
            n = 0 if prev is None else prev.n + 1
            calls.append(("s1", n))
            return V(n=n)

        @step
        async def s2(self, x=depends(s1)) -> V:
            calls.append(("s2", x.n))
            return V(n=x.n)

        @step
        async def s3(self, x=depends(s2)) -> V:
            calls.append(("s3", x.n))
            return V(n=x.n)

        steps = (s1, s2, s3)
        edges = (
            edge(START).to(s1),
            edge(s1).to(s2),
            edge(s2).to(s3),
            edge(s3).when(lambda r: r.n >= 2).to(EXIT).otherwise(s1),
        )

    return ChainStage


# --- cursor failure modes ----------------------------------------------------------


def test_corrupt_cursor_restarts_from_start(run, tmp_path):
    calls = []
    ChainStage = _chain(calls)
    persist = DiskPersistService(base_dir=tmp_path)
    run(ChainStage(persist_service=persist).run_steps())

    (tmp_path / "Chain" / f"{_CURSOR_KEY}.json").write_text("{not json at all")
    calls.clear()
    run(ChainStage(persist_service=persist).run_steps())
    assert calls[0][0] == "s1"          # unreadable checkpoint -> start over, don't crash


def test_stale_cursor_naming_a_gone_step_restarts_from_start(run, tmp_path):
    calls = []
    ChainStage = _chain(calls)
    persist = DiskPersistService(base_dir=tmp_path)
    stage = ChainStage(persist_service=persist)
    stage._save_cursor(_Cursor(next="a_step_from_an_older_graph"))

    run(stage.run_steps())
    assert calls[0][0] == "s1"


def test_unreadable_backend_is_not_swallowed(run, tmp_path):
    """Only a missing or damaged cursor is absorbed. A backend that can't be read is a
    real failure — hiding it would silently re-run every step and repeat side effects."""

    class Broken(DiskPersistService):
        def read(self, key, model):
            if key.endswith(_CURSOR_KEY):
                raise ConnectionError("backend down")
            return super().read(key, model)

    calls = []
    ChainStage = _chain(calls)
    with pytest.raises(ConnectionError, match="backend down"):
        run(ChainStage(persist_service=Broken(base_dir=tmp_path)).run_steps())


def test_cursor_write_failure_never_fails_the_pass(run, tmp_path):
    """Best-effort checkpointing: the steps' own output is the real state."""

    class NoCursorWrites(DiskPersistService):
        def write(self, key, value, model):
            if key.endswith(_CURSOR_KEY):
                raise OSError("read-only")
            return super().write(key, value, model)

    calls = []
    ChainStage = _chain(calls)
    run(ChainStage(persist_service=NoCursorWrites(base_dir=tmp_path)).run_steps())
    assert calls == [("s1", 0), ("s2", 0), ("s3", 0), ("s1", 1), ("s2", 1), ("s3", 1),
                     ("s1", 2), ("s2", 2), ("s3", 2)]


def test_crash_on_the_entry_step_leaves_nothing_to_resume(run, tmp_path):
    class BoomStage(Stage):
        @step
        async def only(self) -> V:
            raise RuntimeError("nope")

        steps = (only,)
        edges = (edge(START).to(only), edge(only).to(EXIT))

    persist = DiskPersistService(base_dir=tmp_path)
    with pytest.raises(RuntimeError, match="nope"):
        run(BoomStage(persist_service=persist).run_steps())
    assert not (tmp_path / "Boom" / f"{_CURSOR_KEY}.json").exists()


def test_start_at_checkpoints_before_running_its_entry(run, tmp_path):
    """A crash in the entered step must not leave an older cursor in place — the next
    plain run would otherwise resume somewhere this run never intended."""
    boom = {"armed": False}

    class EnterStage(Stage):
        @step
        async def a(self, prev=optional_depends("c")) -> V:
            return V(n=0 if prev is None else prev.n + 1)

        @step
        async def b(self, x=depends(a)) -> V:
            if boom["armed"]:
                raise RuntimeError("crash in b")
            return V(n=x.n)

        @step
        async def c(self, x=depends(b)) -> V:
            return V(n=x.n)

        steps = (a, b, c)
        edges = (
            edge(START).to(a),
            edge(a).to(b),
            edge(b).to(c),
            edge(c).when(lambda r: r.n >= 1).to(EXIT).otherwise(a),
        )

    persist = DiskPersistService(base_dir=tmp_path)
    stage = EnterStage(persist_service=persist)
    run(stage.run_steps())
    stage._save_cursor(_Cursor(next="c"))       # some older, unrelated checkpoint

    boom["armed"] = True
    with pytest.raises(RuntimeError, match="crash in b"):
        run(stage.run_steps(start_at="b"))

    cursor = persist.fetch(f"Enter/{_CURSOR_KEY}", _Cursor)
    assert cursor.next == "b"                   # not the stale 'c'


def test_predicate_failure_propagates_and_leaves_the_cursor_put(run, tmp_path):
    class BadPredStage(Stage):
        @step
        async def a(self, prev=optional_depends("b")) -> V:
            return V(n=0)

        @step
        async def b(self, x=depends(a)) -> V:
            return V(n=x.n)

        steps = (a, b)
        edges = (
            edge(START).to(a),
            edge(a).to(b),
            edge(b).when(lambda r: 1 // r.n > 0).to(EXIT).otherwise(a),   # n is 0
        )

    persist = DiskPersistService(base_dir=tmp_path)
    with pytest.raises(ZeroDivisionError):
        run(BadPredStage(persist_service=persist).run_steps())
    assert persist.fetch(f"BadPred/{_CURSOR_KEY}", _Cursor).next == "b"


# --- the fuse ----------------------------------------------------------------------


def test_fuse_allows_exactly_max_steps(run, tmp_path):
    """A graph needing exactly `max_steps` executions completes; one more raises."""

    def build(max_steps):
        class FuseStage(Stage):
            @step
            async def a(self, prev=optional_depends("a")) -> V:
                return V(n=0 if prev is None else prev.n + 1)

            steps = (a,)
            edges = (edge(START).to(a), edge(a).when(lambda r: r.n >= 2).to(EXIT).otherwise(a))

        FuseStage.max_steps = max_steps
        return FuseStage

    # Exits on the 3rd execution (n = 0, 1, 2).
    run(build(3)(persist_service=DiskPersistService(base_dir=tmp_path / "ok")).run_steps())

    with pytest.raises(RuntimeError, match="ran 2 steps without reaching EXIT"):
        run(build(2)(persist_service=DiskPersistService(base_dir=tmp_path / "blown")).run_steps())


# --- interactions ------------------------------------------------------------------


def test_hooks_see_every_pass_and_the_stage_once(run, tmp_path):
    from contextlib import contextmanager

    outputs, stage_entries = [], []

    class Recorder:
        @contextmanager
        def step(self, *, stage_name, step_name, input_type, output_type):
            report = StepReport()
            yield report
            outputs.append((step_name, report.output.n if report.has_output else None))

        @contextmanager
        def stage(self, *, stage_name, step_count):
            stage_entries.append(stage_name)
            yield

    calls = []
    ChainStage = _chain(calls)
    run(ChainStage(persist_service=DiskPersistService(base_dir=tmp_path), hooks=Recorder()).run_steps())

    assert stage_entries == ["Chain"]                       # once per run_steps, not per pass
    assert len(outputs) == 9                                 # every pass reports its output
    assert outputs[-1] == ("s3", 2)


def test_in_memory_backend_resumes_across_stage_instances(run):
    calls = []
    ChainStage = _chain(calls)
    persist = InMemoryPersistService()

    run(ChainStage(persist_service=persist).run_step("s1"))   # seed s1 so s2 can fetch it
    ChainStage(persist_service=persist)._save_cursor(_Cursor(next="s2"))
    calls.clear()

    run(ChainStage(persist_service=persist).run_steps())      # a different instance resumes
    assert calls[0][0] == "s2"


def test_pipeline_resume_lands_under_the_run_id(run, tmp_path):
    calls = []
    ChainStage = _chain(calls)
    p = Pipeline(
        name="pl", run_id="r7", output_root=tmp_path,
        stages={"chain": lambda ps: ChainStage(persist_service=ps)},
    )
    run(p.run(stage="chain"))
    assert (tmp_path / "pl" / "r7" / "Chain" / f"{_CURSOR_KEY}.json").exists()

    p.persist_service.persist(f"Chain/{_CURSOR_KEY}", _Cursor(next="s3"), _Cursor)
    calls.clear()
    run(p.run(stage="chain"))
    assert calls[0][0] == "s3"


def test_run_all_mixes_dag_and_edge_driven_stages(run, tmp_path):
    calls = []
    ChainStage = _chain(calls)

    class AfterStage(Stage):
        @step
        async def summarize(self, x=depends(ChainStage.s3)) -> str:
            return f"chain ended at {x.n}"

        steps = (summarize,)

    p = Pipeline(
        name="mix", output_root=tmp_path,
        stages={
            "chain": lambda ps: ChainStage(persist_service=ps),
            "after": lambda ps: AfterStage(persist_service=ps),
        },
    )
    assert run(p.run(stage="all")) == ["chain ended at 2"]
    assert AfterStage._scheduler is not None and AfterStage._graph is None


def test_follow_edges_on_a_dag_stage_raises_through_the_pipeline(run, tmp_path):
    from _helpers import BStage

    p = Pipeline(name="d", output_root=tmp_path, stages={"b": lambda ps: BStage(persist_service=ps)})
    with pytest.raises(ValueError, match="start_at needs a stage that declares edges"):
        run(p.run(stage="b", step="note", follow_edges=True))


def test_edge_stage_returns_only_what_it_ran(run, tmp_path):
    """A resumed run reports the tail it executed, in declaration order — the same
    partial-result shape a DAG run has when not every step ran."""
    calls = []
    ChainStage = _chain(calls)
    persist = DiskPersistService(base_dir=tmp_path)
    full = run(ChainStage(persist_service=persist).run_steps())
    assert [v.n for v in full] == [2, 2, 2]        # s1, s2, s3 in declaration order

    stage = ChainStage(persist_service=persist)
    stage._save_cursor(_Cursor(next="s3"))
    assert [v.n for v in run(stage.run_steps())] == [2]
