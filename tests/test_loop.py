"""Loops, the cursor, and resume. A flow's whole durable loop state is which node runs
next; everything else a resumed run needs was persisted by the nodes themselves."""

import asyncio
import json

import pytest
from pydantic import BaseModel

from stepper import (
    EXIT,
    START,
    Flow,
    InMemoryPersistService,
    depends,
    edge,
    optional_depends,
    step,
)


class Count(BaseModel):
    rounds: int
    done: bool = False


class Boom(RuntimeError):
    pass


class LoopFlow(Flow[Count]):
    """Counts to three. The counter is a field the step wrote — the framework tracks no
    round number of its own."""

    @step
    async def tick(self, prev: Count | None = optional_depends("check")) -> Count:
        return Count(rounds=(prev.rounds if prev else 0) + 1)

    @step
    async def check(self, c=depends(tick)) -> Count:
        return Count(rounds=c.rounds, done=c.rounds >= 3)

    edges = (
        edge(START).to(tick),
        edge(tick).to(check),
        edge(check).when(lambda c: c.done).to(EXIT).otherwise(tick),
    )


def test_a_loop_runs_until_a_predicate_exits(run, mem):
    assert run(LoopFlow().run(persist_service=mem)) == Count(rounds=3, done=True)


def test_a_step_overwrites_its_key_every_pass(run, mem):
    run(LoopFlow().run(persist_service=mem))
    assert mem.fetch("loop/tick", Count) == Count(rounds=3)


def test_a_finished_loop_marks_its_cursor_done(run, mem):
    run(LoopFlow().run(persist_service=mem))
    assert json.loads(mem._store["loop/_loop_cursor"]) == {"next": None, "done": True}


def test_a_finished_loop_starts_at_start_not_at_the_old_cursor(run, mem):
    """A `done` cursor means the next run begins again rather than resuming a completed
    loop — but it begins against whatever the last run left behind. Here `tick` reads the
    previous run's `check` through its back edge, so it counts on from 3 and exits after
    one pass. A run wanting a clean slate gets a fresh `run_id`."""
    run(LoopFlow().run(persist_service=mem))
    assert run(LoopFlow().run(persist_service=mem)) == Count(rounds=4, done=True)


def test_a_fresh_run_id_gives_a_clean_run(run, mem):
    assert run(LoopFlow().run(run_id="a", persist_service=mem)) == Count(rounds=3, done=True)
    assert run(LoopFlow().run(run_id="b", persist_service=mem)) == Count(rounds=3, done=True)


# --- resume ---------------------------------------------------------------------------


RAN: list[str] = []


class CrashOnce(Flow[Count]):
    """Fails once in `second`. `RAN` records which nodes actually executed, which is the
    only way to tell a resume from a restart — both produce the same value."""

    fail = True

    @step
    async def first(self) -> Count:
        RAN.append("first")
        return Count(rounds=1)

    @step
    async def second(self, c=depends(first)) -> Count:
        RAN.append("second")
        if type(self).fail:
            type(self).fail = False
            raise Boom("half-run")
        return Count(rounds=c.rounds + 1, done=True)

    edges = (edge(START).to(first), edge(first).to(second), edge(second).to(EXIT))


@pytest.fixture
def crashed(mem):
    """A run that died in `second`, with `first` already persisted."""
    CrashOnce.fail = True
    RAN.clear()
    with pytest.raises(Boom):
        asyncio.run(CrashOnce().run(persist_service=mem))
    RAN.clear()
    return mem


def test_a_crash_leaves_the_cursor_on_the_node_that_failed(crashed):
    assert json.loads(crashed._store["crashonce/_loop_cursor"]) == {
        "next": "second",
        "done": False,
    }


def test_a_rerun_resumes_at_the_cursor_without_repeating_what_finished(run, crashed):
    assert run(CrashOnce().run(persist_service=crashed)) == Count(rounds=2, done=True)
    assert RAN == ["second"]                     # `first` was not re-run


def test_an_unreadable_cursor_restarts_from_start(run, crashed):
    crashed._store["crashonce/_loop_cursor"] = b"{not json"
    assert run(CrashOnce().run(persist_service=crashed)) == Count(rounds=2, done=True)
    assert RAN == ["first", "second"]            # the whole graph again


def test_a_cursor_naming_a_node_the_graph_lost_restarts_from_start(run, crashed):
    crashed._store["crashonce/_loop_cursor"] = b'{"next":"gone","done":false}'
    assert run(CrashOnce().run(persist_service=crashed)) == Count(rounds=2, done=True)
    assert RAN == ["first", "second"]


def test_a_missing_cursor_restarts_from_start(run, crashed):
    del crashed._store["crashonce/_loop_cursor"]
    assert run(CrashOnce().run(persist_service=crashed)) == Count(rounds=2, done=True)
    assert RAN == ["first", "second"]


# --- the backend contracts around the cursor ------------------------------------------


def test_a_cursor_that_cannot_be_written_does_not_fail_the_run(run):
    """Checkpointing is an optimization; the nodes' own output is the real state."""

    class NoCursorWrites(InMemoryPersistService):
        def write(self, key, value, model):
            if key.endswith("_loop_cursor"):
                raise OSError("read-only")
            super().write(key, value, model)

    assert run(LoopFlow().run(persist_service=NoCursorWrites())) == Count(rounds=3, done=True)


def test_a_backend_that_cannot_be_read_at_all_propagates(run, crashed):
    """Swallowing this would silently re-run every node from the top, duplicating exactly
    the side effects the resume contract asks callers to guard."""

    class Broken(InMemoryPersistService):
        def read(self, key, model):
            raise PermissionError("backend down")

    broken = Broken()
    broken._store = crashed._store
    with pytest.raises(PermissionError, match="backend down"):
        run(CrashOnce().run(persist_service=broken))


# --- resume inside a nested flow -------------------------------------------------------


def test_a_crash_inside_a_child_leaves_a_cursor_at_every_level(run, mem):
    class Outer(Flow[Count]):
        @step
        async def seed(self) -> Count:
            RAN.append("seed")
            return Count(rounds=0)

        inner = CrashOnce.bind()

        @step
        async def tail(self, c=depends(inner)) -> Count:
            RAN.append("tail")
            return c

        edges = (
            edge(START).to(seed),
            edge(seed).to(inner),
            edge(inner).to(tail),
            edge(tail).to(EXIT),
        )

    CrashOnce.fail = True
    RAN.clear()
    with pytest.raises(Boom):
        run(Outer().run(persist_service=mem))
    assert json.loads(mem._store["outer/_loop_cursor"])["next"] == "inner"
    assert json.loads(mem._store["outer/inner/_loop_cursor"])["next"] == "second"

    RAN.clear()
    assert run(Outer().run(persist_service=mem)) == Count(rounds=2, done=True)
    # The parent resumes at `inner`, which resumes at `second` — `seed` and `first` stay put.
    assert RAN == ["second", "tail"]


# --- entering the graph by hand -------------------------------------------------------


def test_follow_edges_enters_at_a_chosen_node(run, mem):
    root = LoopFlow().mount(persist_service=mem)
    run(root.run())                                    # leaves rounds=3
    # Re-entering at `check` runs check alone, which already sees done.
    assert run(root.run("check", follow_edges=True)) == Count(rounds=3, done=True)


def test_follow_edges_overrides_a_saved_cursor(run, crashed):
    root = CrashOnce().mount(persist_service=crashed)
    # The cursor says `second`; entering at `first` runs first, then follows to second.
    assert run(root.run("first", follow_edges=True)) == Count(rounds=2, done=True)
    assert RAN == ["first", "second"]


def test_follow_edges_needs_a_target(run, mem):
    root = LoopFlow().mount(persist_service=mem)
    with pytest.raises(ValueError, match="needs a target"):
        run(root.run(follow_edges=True))


def test_running_one_node_does_not_move_the_cursor(run, mem):
    root = LoopFlow().mount(persist_service=mem)
    run(root.run())
    before = mem._store["loop/_loop_cursor"]
    run(root.run("tick"))
    assert mem._store["loop/_loop_cursor"] == before


# --- the fuse -------------------------------------------------------------------------


def test_max_steps_stops_a_runaway_predicate(run, mem):
    class Runaway(Flow[Count]):
        max_steps = 5

        @step
        async def spin(self, prev: Count | None = optional_depends("spin")) -> Count:
            return Count(rounds=(prev.rounds if prev else 0) + 1)

        edges = (edge(START).to(spin), edge(spin).when(lambda c: c.done).to(EXIT).otherwise(spin))

    with pytest.raises(RuntimeError, match="ran 5 nodes without reaching EXIT"):
        run(Runaway().run(persist_service=mem))


# --- a nested flow keeps its own cursor -----------------------------------------------


def test_two_mounts_of_a_looping_flow_loop_independently(run, mem):
    class Pair(Flow[Count]):
        left = LoopFlow.bind()
        right = LoopFlow.bind()

        @step
        async def total(self, a=depends(left), b=depends(right)) -> Count:
            return Count(rounds=a.rounds + b.rounds, done=True)

        edges = (
            edge(START).to(left),
            edge(left).to(right),
            edge(right).to(total),
            edge(total).to(EXIT),
        )

    assert run(Pair().run(persist_service=mem)) == Count(rounds=6, done=True)
    assert "pair/left/_loop_cursor" in mem._store
    assert "pair/right/_loop_cursor" in mem._store
