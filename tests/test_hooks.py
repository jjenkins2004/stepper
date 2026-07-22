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


class TaggedHooks:
    """Like `TimelineHooks`, but tags each entry with a hook id so a shared timeline
    pins down fan-out order across two hooks (enter forward, exit reverse)."""

    def __init__(self, tag: str, timeline: list[str]):
        self.tag = tag
        self.timeline = timeline

    @contextmanager
    def step(self, *, stage_name, step_name, input_type, output_type):
        self.timeline.append(f"{self.tag}:step-before:{step_name}")
        try:
            yield
        except Exception:
            self.timeline.append(f"{self.tag}:step-error:{step_name}")
            raise
        else:
            self.timeline.append(f"{self.tag}:step-after:{step_name}")

    @contextmanager
    def stage(self, *, stage_name, step_count):
        self.timeline.append(f"{self.tag}:stage-before:{stage_name}")
        try:
            yield
        finally:
            self.timeline.append(f"{self.tag}:stage-after:{stage_name}")


def test_empty_hooks_list_is_noop(persist, run):
    # [] normalizes to () — no hooks entered, step runs and persists as usual.
    results = run(AStage(persist_service=persist, hooks=[]).run_steps())
    assert results == [Item(value=1)]


def test_two_hooks_fan_out_enter_forward_exit_reverse(persist, run):
    """Both hooks bracket the stage and the step; entered in list order, exited in
    reverse (ExitStack default)."""
    timeline: list[str] = []

    @step
    async def work(self) -> Item:
        timeline.append("body:work")
        return Item(value=1)

    class WorkStage(Stage):
        steps = (work,)

    a, b = TaggedHooks("A", timeline), TaggedHooks("B", timeline)
    run(WorkStage(persist_service=persist, hooks=[a, b]).run_steps())
    assert timeline == [
        "A:stage-before:Work", "B:stage-before:Work",   # enter forward
        "A:step-before:work", "B:step-before:work",     # enter forward
        "body:work",
        "B:step-after:work", "A:step-after:work",       # exit reverse
        "B:stage-after:Work", "A:stage-after:Work",     # exit reverse
    ]


def test_two_hooks_both_receive_step_output(persist, run):
    """Each hook's own StepReport is filled with the step's return value — no middleman."""
    cap_a: dict = {}
    cap_b: dict = {}
    run(AStage(persist_service=persist, hooks=[_OutputHooks(cap_a), _OutputHooks(cap_b)]).run_step("prod"))
    for cap in (cap_a, cap_b):
        assert cap["has_output"] is True
        assert cap["output"] == Item(value=1)


def test_hook_yielding_no_report_is_tolerated(persist, run):
    """A hook that yields nothing sits alongside one that yields a StepReport; the
    report-yielding hook still receives the output, the other is simply skipped."""
    cap: dict = {}

    class NoReportHooks:
        @contextmanager
        def step(self, *, stage_name, step_name, input_type, output_type):
            yield  # yields None — no StepReport to fill
        @contextmanager
        def stage(self, *, stage_name, step_count):
            yield

    run(AStage(persist_service=persist, hooks=[NoReportHooks(), _OutputHooks(cap)]).run_step("prod"))
    assert cap["has_output"] is True
    assert cap["output"] == Item(value=1)


def test_two_hooks_both_see_step_failure_and_it_reraises(persist, run):
    """A raising step body propagates into every entered hook at its yield (reverse
    order), and the exception re-raises out of the run."""
    timeline: list[str] = []

    @step
    async def boom(self) -> Item:
        timeline.append("body:boom")
        raise RuntimeError("kaboom")

    class BoomStage(Stage):
        steps = (boom,)

    a, b = TaggedHooks("A", timeline), TaggedHooks("B", timeline)
    with pytest.raises(RuntimeError, match="kaboom"):
        run(BoomStage(persist_service=persist, hooks=[a, b]).run_step("boom"))
    assert timeline == [
        "A:step-before:boom", "B:step-before:boom",
        "body:boom",
        "B:step-error:boom", "A:step-error:boom",  # both see it, reverse order; no step-after
    ]


def test_hook_enter_failure_unwinds_entered_hooks(persist, run):
    """The sharp edge: a hook raising in __enter__ unwinds the already-entered hooks with
    that exception (they see it at their yield), the step body never runs, and it re-raises."""
    timeline: list[str] = []

    @step
    async def work(self) -> Item:
        timeline.append("body:work")
        return Item(value=1)

    class WorkStage(Stage):
        steps = (work,)

    class BadEnter:
        @contextmanager
        def step(self, *, stage_name, step_name, input_type, output_type):
            timeline.append("B:enter-boom")
            raise RuntimeError("enter-boom")
            yield  # unreachable — makes step() a generator context manager
        @contextmanager
        def stage(self, *, stage_name, step_count):
            yield

    good = TaggedHooks("A", timeline)
    with pytest.raises(RuntimeError, match="enter-boom"):
        run(WorkStage(persist_service=persist, hooks=[good, BadEnter()]).run_step("work"))
    assert timeline == [
        "A:step-before:work",  # first hook entered
        "B:enter-boom",        # second hook blew up in __enter__
        "A:step-error:work",   # first hook unwound with that exception
    ]
    assert "body:work" not in timeline  # step never ran


def test_hook_exit_failure_propagates_to_other_hook(persist, run):
    """The sharp edge, exit side: a hook raising in __exit__ propagates into the
    remaining entered hooks (reverse order) and re-raises."""
    timeline: list[str] = []

    @step
    async def work(self) -> Item:
        timeline.append("body:work")
        return Item(value=1)

    class WorkStage(Stage):
        steps = (work,)

    class BadExit:
        @contextmanager
        def step(self, *, stage_name, step_name, input_type, output_type):
            timeline.append("B:step-before:work")
            yield
            timeline.append("B:exit-boom")
            raise RuntimeError("exit-boom")
        @contextmanager
        def stage(self, *, stage_name, step_count):
            yield

    good = TaggedHooks("A", timeline)
    with pytest.raises(RuntimeError, match="exit-boom"):
        run(WorkStage(persist_service=persist, hooks=[good, BadExit()]).run_step("work"))
    assert timeline == [
        "A:step-before:work", "B:step-before:work",
        "body:work",
        "B:exit-boom",         # last-entered exits first and raises
        "A:step-error:work",   # exception thrown into the remaining hook
    ]


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


def test_pipeline_applies_a_list_of_hooks_to_all_stages(tmp_path, run):
    """A sequence handed to the Pipeline is normalized and both hooks wrap every stage/step."""
    timeline: list[str] = []
    p = Pipeline(
        name="p",
        run_id="r",
        output_root=tmp_path,
        hooks=[TaggedHooks("A", timeline), TaggedHooks("B", timeline)],
        stages={
            "a": lambda ps: AStage(persist_service=ps),
            "b": lambda ps: BStage(persist_service=ps),
        },
    )
    run(p.run(module="all"))
    for event in (
        "A:stage-before:A", "B:stage-before:A", "A:stage-after:A", "B:stage-after:A",
        "A:step-before:prod", "B:step-before:prod", "A:step-after:prod", "B:step-after:prod",
        "A:step-before:note", "B:step-before:note",
    ):
        assert event in timeline
