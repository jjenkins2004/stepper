"""Hooks wrap every node a run executes. The framework only ever touches its own
`StepReport`, so nothing here couples it to a tracing library."""

from contextlib import contextmanager

import pytest
from _helpers import Item, RootFlow

from stepper import EXIT, START, Flow, StepReport, edge, step


class Recorder:
    """Records the order things are entered and exited, and each step's output."""

    def __init__(self, tag: str = "h") -> None:
        self.tag = tag
        self.events: list[str] = []
        self.outputs: list[object] = []

    @contextmanager
    def step(self, *, path: str, input_type: str, output_type: str):
        report = StepReport()
        self.events.append(f"{self.tag}:enter step {path}({input_type})->{output_type}")
        yield report
        self.events.append(f"{self.tag}:exit step {path}")
        self.outputs.append(report.output if report.has_output else None)

    @contextmanager
    def flow(self, *, path: str, node_count: int):
        self.events.append(f"{self.tag}:enter flow {path}[{node_count}]")
        yield
        self.events.append(f"{self.tag}:exit flow {path}")


class Silent:
    """A hook that yields nothing — the framework must cope."""

    def __init__(self) -> None:
        self.steps = 0

    @contextmanager
    def step(self, *, path: str, input_type: str, output_type: str):
        self.steps += 1
        yield None

    @contextmanager
    def flow(self, *, path: str, node_count: int):
        yield


class Boom(RuntimeError):
    pass


class Failing(Flow[Item]):
    @step
    async def bad(self) -> Item:
        raise Boom("nope")

    edges = (edge(START).to(bad), edge(bad).to(EXIT))


def test_a_step_hook_sees_the_path_and_the_types(run, mem):
    rec = Recorder()
    run(RootFlow().run(persist_service=mem, hooks=rec))
    assert "h:enter step root/start(None)->Item" in rec.events
    assert "h:enter step root/total(Item, Item)->Item" in rec.events


def test_a_flow_hook_wraps_every_flow_including_nested(run, mem):
    rec = Recorder()
    run(RootFlow().run(persist_service=mem, hooks=rec))
    entered = [e for e in rec.events if "enter flow" in e]
    assert entered == [
        "h:enter flow root[4]",
        "h:enter flow root/left[1]",
        "h:enter flow root/right[1]",
    ]


def test_the_report_carries_the_output_after_the_step_ran(run, mem):
    rec = Recorder()
    run(RootFlow().run(persist_service=mem, hooks=rec))
    assert Item(value=1) in rec.outputs
    assert Item(value=6) in rec.outputs


def test_a_step_that_persists_nothing_leaves_the_report_empty(run, mem):
    rec = Recorder()

    class Quiet(Flow):
        @step
        async def nothing(self) -> None:
            return None

        edges = (edge(START).to(nothing), edge(nothing).to(EXIT))

    run(Quiet().run(persist_service=mem, hooks=rec))
    assert rec.outputs == [None]


def test_several_hooks_enter_in_order_and_exit_in_reverse(run, mem):
    a, b = Recorder("a"), Recorder("b")
    shared: list[str] = []
    a.events = b.events = shared
    run(RootFlow().run(persist_service=mem, hooks=[a, b]))
    start = shared.index("a:enter step root/start(None)->Item")
    assert shared[start : start + 4] == [
        "a:enter step root/start(None)->Item",
        "b:enter step root/start(None)->Item",
        "b:exit step root/start",
        "a:exit step root/start",
    ]


def test_a_hook_may_yield_nothing(run, mem):
    silent = Silent()
    run(RootFlow().run(persist_service=mem, hooks=silent))
    assert silent.steps == 4


def test_a_lone_hook_and_a_one_element_list_behave_the_same(run, mem):
    one, listed = Recorder(), Recorder()
    run(RootFlow().run(persist_service=mem, hooks=one))
    run(RootFlow().run(persist_service=mem, hooks=[listed]))
    assert one.events == listed.events


def test_no_hooks_is_a_clean_no_op(run, mem):
    assert run(RootFlow().run(persist_service=mem)) == Item(value=6)


def test_a_failing_step_skips_the_after_yield_code(run, mem):
    rec = Recorder()
    with pytest.raises(Boom):
        run(Failing().run(persist_service=mem, hooks=rec))
    assert any("enter step" in e for e in rec.events)
    assert not any("exit step" in e for e in rec.events)


def test_a_hook_can_observe_a_failure_with_try_finally(run, mem):
    seen: list[str] = []

    class Watcher:
        @contextmanager
        def step(self, *, path: str, input_type: str, output_type: str):
            try:
                yield StepReport()
            except Boom:
                seen.append(f"failed {path}")
                raise

        @contextmanager
        def flow(self, *, path: str, node_count: int):
            yield

    with pytest.raises(Boom):
        run(Failing().run(persist_service=mem, hooks=Watcher()))
    assert seen == ["failed failing/bad"]


def test_hooks_reach_every_depth(run, mem):
    rec = Recorder()
    run(RootFlow().run(persist_service=mem, hooks=rec))
    assert "h:enter step root/left/double(Item)->Item" in rec.events
    assert "h:enter step root/right/double(Item)->Item" in rec.events
