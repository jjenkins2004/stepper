"""Hooks — user-supplied context managers wrap each step and stage run.

The key behaviors under test: the code *before* a hook's `yield` runs before the step
body, and the code *after* the `yield` runs after the body returns and its output is
persisted. Steps and hooks append to one shared `timeline`, so a single list assertion
pins down the exact interleaving.
"""

from contextlib import contextmanager, nullcontext

import pytest

from stepper import Pipeline, StepReport
from stepper.stage import Stage
from stepper.step import depends, step

from _helpers import AStage, BStage, Item


class TimelineHooks:
    """Records `stage-before/after` and `step-before/after/error` into a shared
    timeline. Step bodies append `body:<name>` to the same list, so the assertion
    shows hooks bracketing the real work."""

    def __init__(self, timeline: list[str]):
        self.timeline = timeline

    @contextmanager
    def step(self, *, stage_name, step_name, input_type, output_type):
        self.timeline.append(f"step-before:{step_name}")
        try:
            yield
        except Exception:
            self.timeline.append(f"step-error:{step_name}")
            raise
        else:
            self.timeline.append(f"step-after:{step_name}")

    @contextmanager
    def stage(self, *, stage_name, step_count):
        self.timeline.append(f"stage-before:{stage_name}")
        yield
        self.timeline.append(f"stage-after:{stage_name}")


def test_step_hook_brackets_body_and_stage_wraps_step(persist, run):
    """before-yield -> step body -> after-yield, with the stage hook around it all."""
    timeline: list[str] = []

    @step
    async def work(self) -> Item:
        timeline.append("body:work")
        return Item(value=1)

    class WorkStage(Stage):
        steps = (work,)

    run(WorkStage(persist_service=persist, hooks=TimelineHooks(timeline)).run_steps())
    assert timeline == [
        "stage-before:Work",
        "step-before:work",
        "body:work",       # before-yield ran before the body...
        "step-after:work",  # ...and after-yield ran after it
        "stage-after:Work",
    ]


def test_hooks_bracket_each_step_in_dep_order(persist, run):
    """Every step is individually bracketed, in dependency order (a before b)."""
    timeline: list[str] = []

    @step
    async def a(self) -> Item:
        timeline.append("body:a")
        return Item(value=1)

    @step
    async def b(self, up=depends(a)) -> Item:
        timeline.append("body:b")
        return Item(value=up.value + 1)

    class ChainStage(Stage):
        steps = (a, b)

    run(ChainStage(persist_service=persist, hooks=TimelineHooks(timeline)).run_steps())
    assert timeline == [
        "stage-before:Chain",
        "step-before:a", "body:a", "step-after:a",
        "step-before:b", "body:b", "step-after:b",
        "stage-after:Chain",
    ]


class _PersistProbeHooks:
    """Records whether the step's output file exists before vs after the `yield`."""

    def __init__(self, base_dir, saw: dict[str, bool]):
        self._base = base_dir
        self._saw = saw

    @contextmanager
    def step(self, *, stage_name, step_name, input_type, output_type):
        path = self._base / stage_name / f"{step_name}.json"
        self._saw["before"] = path.exists()
        yield
        self._saw["after"] = path.exists()

    def stage(self, *, stage_name, step_count):
        return nullcontext()


def test_after_yield_runs_after_output_is_persisted(persist, tmp_path, run):
    """Proves ordering against real side effects: before-yield sees nothing on disk;
    after-yield sees the persisted output (fetch -> run -> persist all happened inside
    the `with`, before the context manager exits)."""
    saw: dict[str, bool] = {}
    run(AStage(persist_service=persist, hooks=_PersistProbeHooks(tmp_path, saw)).run_step("prod"))
    assert saw["before"] is False  # nothing persisted yet when the step body starts
    assert saw["after"] is True    # output on disk by the time after-yield runs


class _OutputHooks:
    """Yields a `StepReport` and, after the yield, stashes what the framework filled."""

    def __init__(self, captured: dict):
        self._captured = captured

    @contextmanager
    def step(self, *, stage_name, step_name, input_type, output_type):
        report = StepReport()
        yield report
        self._captured["has_output"] = report.has_output
        self._captured["output"] = report.output

    @contextmanager
    def stage(self, *, stage_name, step_count):
        yield


def test_step_report_receives_the_step_output(persist, run):
    """The framework fills the yielded StepReport with the step's return value, readable
    in the after-yield code."""
    captured: dict = {}
    run(AStage(persist_service=persist, hooks=_OutputHooks(captured)).run_step("prod"))
    assert captured["has_output"] is True
    assert captured["output"] == Item(value=1)  # AStage.prod returns Item(value=1)


def test_step_report_has_no_output_for_modelless_step(persist, run):
    """A step with no return annotation persists nothing, so its StepReport stays empty."""
    captured: dict = {}

    @step
    async def noop(self):  # no return annotation -> model is None
        return "ignored"

    class NoopStage(Stage):
        steps = (noop,)

    run(NoopStage(persist_service=persist, hooks=_OutputHooks(captured)).run_step("noop"))
    assert captured["has_output"] is False
    assert captured["output"] is None


def test_step_hook_sees_error_and_skips_after(persist, run):
    """A raising step body: before-yield and the body ran, the error branch fired, the
    after-yield (`else`) did not, and the exception propagates through the hook."""
    timeline: list[str] = []

    @step
    async def boom(self) -> Item:
        timeline.append("body:boom")
        raise RuntimeError("kaboom")

    class BoomStage(Stage):
        steps = (boom,)

    with pytest.raises(RuntimeError, match="kaboom"):
        run(BoomStage(persist_service=persist, hooks=TimelineHooks(timeline)).run_step("boom"))
    assert timeline == ["step-before:boom", "body:boom", "step-error:boom"]  # no step-after


def test_default_hooks_are_noop(persist, run):
    # no hooks passed -> NoOpHooks; runs fine and persists as usual
    results = run(AStage(persist_service=persist).run_steps())
    assert results == [Item(value=1)]


def test_pipeline_applies_hooks_to_all_stages(tmp_path, run):
    timeline: list[str] = []
    p = Pipeline(
        name="p",
        run_id="r",
        output_root=tmp_path,
        hooks=TimelineHooks(timeline),
        stages={
            "a": lambda ps: AStage(persist_service=ps),
            "b": lambda ps: BStage(persist_service=ps),
        },
    )
    run(p.run(module="all"))
    # every stage and step the pipeline ran is wrapped (order across concurrent steps
    # within a stage may interleave, so assert membership).
    for event in (
        "stage-before:A", "stage-after:A",
        "stage-before:B", "stage-after:B",
        "step-before:prod", "step-after:prod",
        "step-before:consume", "step-after:consume",
        "step-before:note", "step-after:note",
    ):
        assert event in timeline
