"""Fan-out and fan-in: `edge(a).to(b, c)` runs the arms at once, `edge(b, c).to(j)` brings
them back.

The first half is about graphs that must *never* reach a run. A fan-out is a structured
region — source, disjoint arms, one derived join — and every way of writing something that
isn't one raises while the class is being created. The second half runs the good ones.
"""

import asyncio

import pytest
from pydantic import BaseModel

from stepper import (
    EXIT,
    START,
    Flow,
    depends,
    edge,
    optional_depends,
    require,
    step,
)


class Item(BaseModel):
    value: int = 0
    tag: str = ""


def _flow(**body):
    """Build a Flow subclass from a dict body, so a bad graph raises here rather than at
    import time."""
    return type("Made", (Flow,), body)


# --- the builder, before any flow exists -----------------------------------------------


def test_edge_with_no_sources_is_rejected():
    with pytest.raises(TypeError, match="at least one source"):
        edge()


def test_to_with_no_targets_is_rejected():
    with pytest.raises(TypeError, match="at least one target"):
        edge(START).to()


def test_otherwise_with_no_targets_is_rejected():
    e = edge(START).when(lambda r: True).to(EXIT)
    with pytest.raises(TypeError, match="at least one target"):
        e.otherwise()


def test_a_multi_source_edge_cannot_branch():
    with pytest.raises(TypeError, match="routes unconditionally"):
        edge(START, EXIT).when(lambda r: True)


def test_a_multi_source_edge_cannot_also_fan_out():
    with pytest.raises(TypeError, match="can't also fan out"):
        edge(START, EXIT).to(START, EXIT)


def test_resolve_refuses_to_pick_one_target_out_of_a_fan_out():
    e = edge(START).to(START, EXIT)
    with pytest.raises(TypeError, match="no single target"):
        e.resolve(None)


# --- bad graphs: every one raises when the class is created -----------------------------


def test_start_cannot_fan_out():
    @step
    async def a(self) -> Item: ...

    @step
    async def b(self) -> Item: ...

    @step
    async def j(self, x=depends(a), y=depends(b)) -> Item: ...

    with pytest.raises(TypeError, match=r"edge\(START\) has one target"):
        _flow(
            a=a, b=b, j=j,
            edges=(edge(START).to(a, b), edge(a, b).to(j), edge(j).to(EXIT)),
        )


def test_start_cannot_share_an_edge_with_a_node():
    @step
    async def a(self) -> Item: ...

    with pytest.raises(TypeError, match="edge.START. names START alone"):
        _flow(a=a, edges=(edge(START, a).to(a), edge(a).to(EXIT)))


def test_a_fan_out_cannot_have_exit_as_an_arm():
    @step
    async def a(self) -> Item: ...

    @step
    async def b(self, x=depends(a)) -> Item: ...

    with pytest.raises(TypeError, match="fans out to EXIT"):
        _flow(a=a, b=b, edges=(edge(START).to(a), edge(a).to(b, EXIT), edge(b).to(EXIT)))


def test_the_same_node_twice_in_one_fan_out_is_rejected():
    @step
    async def a(self) -> Item: ...

    @step
    async def b(self, x=depends(a)) -> Item: ...

    with pytest.raises(TypeError, match="the same node twice"):
        _flow(a=a, b=b, edges=(edge(START).to(a), edge(a).to(b, b), edge(b).to(EXIT)))


def test_arms_that_only_meet_at_exit_are_rejected():
    @step
    async def a(self) -> Item: ...

    @step
    async def b(self, x=depends(a)) -> Item: ...

    @step
    async def c(self, x=depends(a)) -> Item: ...

    with pytest.raises(TypeError, match="never meet"):
        _flow(a=a, b=b, c=c, edges=(edge(START).to(a), edge(a).to(b, c), edge(b, c).to(EXIT)))


def test_arms_that_diverge_forever_are_rejected():
    """Each arm has its own tail; they never come back together."""

    @step
    async def a(self) -> Item: ...

    @step
    async def b(self, x=depends(a)) -> Item: ...

    @step
    async def c(self, x=depends(a)) -> Item: ...

    @step
    async def b_tail(self, x=depends(b)) -> Item: ...

    @step
    async def c_tail(self, x=depends(c)) -> Item: ...

    with pytest.raises(TypeError, match="never meet"):
        _flow(
            a=a, b=b, c=c, b_tail=b_tail, c_tail=c_tail,
            edges=(
                edge(START).to(a),
                edge(a).to(b, c),
                edge(b).to(b_tail),
                edge(c).to(c_tail),
                edge(b_tail, c_tail).to(EXIT),
            ),
        )


def test_an_arm_that_is_also_the_join_is_rejected():
    """`a -> (b, c)` plus `b -> c` puts the arms in sequence, not in parallel: `c` would be
    both an arm and where the arms meet, and would run twice."""

    @step
    async def a(self) -> Item: ...

    @step
    async def b(self, x=depends(a)) -> Item: ...

    @step
    async def c(self, x=depends(a)) -> Item: ...

    @step
    async def j(self, x=depends(c)) -> Item: ...

    with pytest.raises(TypeError, match="also where its arms meet"):
        _flow(
            a=a, b=b, c=c, j=j,
            edges=(edge(START).to(a), edge(a).to(b, c), edge(b).to(c), edge(c).to(j), edge(j).to(EXIT)),
        )


def test_an_arm_cannot_loop_back_to_the_node_that_fanned_out():
    @step
    async def a(self, prev: Item = optional_depends("b")) -> Item: ...

    @step
    async def b(self, x=depends(a)) -> Item: ...

    @step
    async def c(self, x=depends(a)) -> Item: ...

    @step
    async def j(self, x=depends(c)) -> Item: ...

    with pytest.raises(TypeError, match="routes back to 'a'"):
        _flow(
            a=a, b=b, c=c, j=j,
            edges=(
                edge(START).to(a),
                edge(a).to(b, c),
                edge(b).when(lambda r: r.value > 0).to(j).otherwise(a),
                edge(c).to(j),
                edge(j).to(EXIT),
            ),
        )


def test_nothing_may_jump_into_the_middle_of_an_arm():
    @step
    async def a(self) -> Item: ...

    @step
    async def b(self, x=depends(a)) -> Item: ...

    @step
    async def c(self, x=depends(a)) -> Item: ...

    @step
    async def j(self, x=depends(b), y=depends(c)) -> Item: ...

    with pytest.raises(TypeError, match="is inside the .* arm of the fan-out"):
        _flow(
            a=a, b=b, c=c, j=j,
            edges=(
                edge(START).to(a),
                edge(a).to(b, c),
                edge(b, c).to(j),
                edge(j).when(lambda r: r.value > 0).to(EXIT).otherwise(b),
            ),
        )


def test_arms_may_not_share_a_node():
    """`b` can reach `shared` or skip it, so `shared` sits in both arms and would run
    twice."""

    @step
    async def a(self) -> Item: ...

    @step
    async def b(self, x=depends(a)) -> Item: ...

    @step
    async def c(self, x=depends(a)) -> Item: ...

    @step
    async def shared(self, x=depends("c")) -> Item: ...

    @step
    async def j(self, x=depends(b)) -> Item: ...

    with pytest.raises(TypeError, match="both reach shared"):
        _flow(
            a=a, b=b, c=c, shared=shared, j=j,
            edges=(
                edge(START).to(a),
                edge(a).to(b, c),
                edge(b).when(lambda r: r.value > 0).to(shared).otherwise(j),
                edge(c).to(shared),
                edge(shared).to(j),
                edge(j).to(EXIT),
            ),
        )


def test_the_source_cannot_also_reach_an_arm_outside_the_fan_out():
    """`common` runs either way — but on `.otherwise` it runs *alone*, so the join would be
    reached with `extra` never having run. Every local check passes: `common`'s only
    predecessor is `check` on both routes."""

    @step
    async def check(self) -> Item: ...

    @step
    async def extra(self, x=depends(check)) -> Item: ...

    @step
    async def common(self, x=depends(check)) -> Item: ...

    @step
    async def j(self, p=depends(extra), q=depends(common)) -> Item: ...

    with pytest.raises(TypeError, match="also routes to common outside its fan-out"):
        _flow(
            check=check, extra=extra, common=common, j=j,
            edges=(
                edge(START).to(check),
                edge(check).when(lambda r: r.value == 0).to(extra, common).otherwise(common),
                edge(extra, common).to(j),
                edge(j).to(EXIT),
            ),
        )


def test_the_source_cannot_side_door_into_the_middle_of_an_arm():
    """The `.otherwise` route enters arm `a` at `m`, skipping both `a` and `b`."""

    @step
    async def s(self) -> Item: ...

    @step
    async def a(self, x=depends(s)) -> Item: ...

    @step
    async def b(self, x=depends(s)) -> Item: ...

    @step
    async def m(self, x: Item = optional_depends("a")) -> Item: ...

    @step
    async def j(self, p=depends(m), q=depends(b)) -> Item: ...

    with pytest.raises(TypeError, match="'s' routes to 'm', which is inside"):
        _flow(
            s=s, a=a, b=b, m=m, j=j,
            edges=(
                edge(START).to(s),
                edge(s).when(lambda r: r.value == 0).to(a, b).otherwise(m),
                edge(a).to(m),
                edge(m, b).to(j),
                edge(j).to(EXIT),
            ),
        )


def test_a_node_cannot_be_an_arm_of_two_different_fan_outs():
    """Deliberately conservative: `a` would be reachable other than through the fan-out
    whose join is being reasoned about."""

    @step
    async def x(self) -> Item: ...

    @step
    async def a(self, i=depends(x)) -> Item: ...

    @step
    async def b(self, i=depends(x)) -> Item: ...

    @step
    async def c(self, i=depends(x)) -> Item: ...

    @step
    async def j(self, p=depends(a)) -> Item: ...

    with pytest.raises(TypeError, match="also routes to a outside its fan-out"):
        _flow(
            x=x, a=a, b=b, c=c, j=j,
            edges=(
                edge(START).to(x),
                edge(x).when(lambda r: r.value == 0).to(a, b).otherwise(a, c),
                edge(a, b, c).to(j),
                edge(j).to(EXIT),
            ),
        )


def test_the_graph_cannot_begin_inside_an_arm():
    """START is nobody's predecessor, so this is the one way into an arm that the
    predecessor check can't see."""

    @step
    async def e(self, prev: Item = optional_depends("s")) -> Item: ...

    @step
    async def b(self, x=depends("s")) -> Item: ...

    @step
    async def j(self, p=depends(e), q=depends(b)) -> Item: ...

    @step
    async def s(self, x=depends(j)) -> Item: ...

    with pytest.raises(TypeError, match="the graph begins at 'e', which is inside"):
        _flow(
            e=e, b=b, j=j, s=s,
            edges=(
                edge(START).to(e),
                edge(e, b).to(j),
                edge(j).when(lambda r: r.value > 0).to(EXIT).otherwise(s),
                edge(s).to(e, b),
            ),
        )


def test_the_graph_cannot_begin_partway_through_an_arm():
    @step
    async def m(self, prev: Item = optional_depends("a")) -> Item: ...

    @step
    async def a(self, x=depends("s")) -> Item: ...

    @step
    async def b(self, x=depends("s")) -> Item: ...

    @step
    async def j(self, p=depends(m), q=depends(b)) -> Item: ...

    @step
    async def s(self, x=depends(j)) -> Item: ...

    with pytest.raises(TypeError, match="the graph begins at 'm', which is inside"):
        _flow(
            m=m, a=a, b=b, j=j, s=s,
            edges=(
                edge(START).to(m),
                edge(a).to(m),
                edge(m, b).to(j),
                edge(j).when(lambda r: r.value > 0).to(EXIT).otherwise(s),
                edge(s).to(a, b),
            ),
        )


def test_one_arm_cannot_depend_on_another():
    @step
    async def a(self) -> Item: ...

    @step
    async def b(self, x=depends(a)) -> Item: ...

    @step
    async def c(self, x=depends("b")) -> Item: ...

    @step
    async def j(self, x=depends(b), y=depends(c)) -> Item: ...

    with pytest.raises(TypeError, match="different arms of the fan-out"):
        _flow(
            a=a, b=b, c=c, j=j,
            edges=(edge(START).to(a), edge(a).to(b, c), edge(b, c).to(j), edge(j).to(EXIT)),
        )


def test_one_arm_cannot_optionally_depend_on_another():
    @step
    async def a(self) -> Item: ...

    @step
    async def b(self, x=depends(a)) -> Item: ...

    @step
    async def c(self, x=optional_depends("b")) -> Item: ...

    @step
    async def j(self, x=depends(b), y=depends(c)) -> Item: ...

    with pytest.raises(TypeError, match="different arms of the fan-out"):
        _flow(
            a=a, b=b, c=c, j=j,
            edges=(edge(START).to(a), edge(a).to(b, c), edge(b, c).to(j), edge(j).to(EXIT)),
        )


def test_a_deep_node_in_one_arm_cannot_depend_on_a_deep_node_in_another():
    @step
    async def a(self) -> Item: ...

    @step
    async def b1(self, x=depends(a)) -> Item: ...

    @step
    async def b2(self, x=depends(b1)) -> Item: ...

    @step
    async def c1(self, x=depends(a)) -> Item: ...

    @step
    async def c2(self, x=depends(c1), sneaky=depends("b2")) -> Item: ...

    @step
    async def j(self, x=depends(b2), y=depends(c2)) -> Item: ...

    with pytest.raises(TypeError, match="different arms of the fan-out"):
        _flow(
            a=a, b1=b1, b2=b2, c1=c1, c2=c2, j=j,
            edges=(
                edge(START).to(a),
                edge(a).to(b1, c1),
                edge(b1).to(b2),
                edge(c1).to(c2),
                edge(b2, c2).to(j),
                edge(j).to(EXIT),
            ),
        )


def test_a_multi_source_edge_still_hits_one_edge_per_node():
    @step
    async def a(self) -> Item: ...

    @step
    async def b(self, x=depends(a)) -> Item: ...

    @step
    async def j(self, x=depends(a), y=depends(b)) -> Item: ...

    with pytest.raises(TypeError, match="more than one edge"):
        _flow(
            a=a, b=b, j=j,
            edges=(edge(START).to(a), edge(a).to(b, j), edge(a, b).to(j), edge(j).to(EXIT)),
        )


def test_a_node_after_the_join_cannot_be_read_from_inside_an_arm():
    """Nothing changes for a plain back-edge read: it still has to be optional."""

    @step
    async def a(self) -> Item: ...

    @step
    async def b(self, x=depends(a), later=depends("j")) -> Item: ...

    @step
    async def c(self, x=depends(a)) -> Item: ...

    @step
    async def j(self, x=depends(b), y=depends(c)) -> Item: ...

    with pytest.raises(TypeError, match="not guaranteed to have run"):
        _flow(
            a=a, b=b, c=c, j=j,
            edges=(edge(START).to(a), edge(a).to(b, c), edge(b, c).to(j), edge(j).to(EXIT)),
        )


def test_terminals_must_still_agree_when_a_fan_out_is_involved():
    class Other(BaseModel):
        n: int

    @step
    async def a(self) -> Item: ...

    @step
    async def b(self, x=depends(a)) -> Item: ...

    @step
    async def c(self, x=depends(a)) -> Item: ...

    @step
    async def j(self, x=depends(b), y=depends(c)) -> Item: ...

    @step
    async def other(self, x=depends(j)) -> Other: ...

    with pytest.raises(TypeError, match="don't agree on what this flow produces"):
        _flow(
            a=a, b=b, c=c, j=j, other=other,
            edges=(
                edge(START).to(a),
                edge(a).to(b, c),
                edge(b, c).to(j),
                edge(j).when(lambda r: r.value > 0).to(EXIT).otherwise(other),
                edge(other).to(EXIT),
            ),
        )


# --- good graphs: shapes that must be accepted ------------------------------------------


def test_the_join_may_depend_on_every_arm():
    """The whole point. Plain intersection over predecessors would reject this."""

    @step
    async def a(self) -> Item: ...

    @step
    async def b(self, x=depends(a)) -> Item: ...

    @step
    async def c(self, x=depends(a)) -> Item: ...

    @step
    async def d(self, x=depends(a)) -> Item: ...

    @step
    async def j(self, p=depends(b), q=depends(c), r=depends(d)) -> Item: ...

    _flow(
        a=a, b=b, c=c, d=d, j=j,
        edges=(edge(START).to(a), edge(a).to(b, c, d), edge(b, c, d).to(j), edge(j).to(EXIT)),
    )


def test_a_node_past_the_join_may_depend_on_an_arm():
    @step
    async def a(self) -> Item: ...

    @step
    async def b(self, x=depends(a)) -> Item: ...

    @step
    async def c(self, x=depends(a)) -> Item: ...

    @step
    async def j(self, p=depends(b), q=depends(c)) -> Item: ...

    @step
    async def after(self, p=depends(j), q=depends(b)) -> Item: ...

    _flow(
        a=a, b=b, c=c, j=j, after=after,
        edges=(
            edge(START).to(a),
            edge(a).to(b, c),
            edge(b, c).to(j),
            edge(j).to(after),
            edge(after).to(EXIT),
        ),
    )


def test_a_fan_out_nested_in_an_arm_may_share_the_outer_join():
    @step
    async def a(self) -> Item: ...

    @step
    async def b(self, x=depends(a)) -> Item: ...

    @step
    async def b1(self, x=depends(b)) -> Item: ...

    @step
    async def b2(self, x=depends(b)) -> Item: ...

    @step
    async def c(self, x=depends(a)) -> Item: ...

    @step
    async def j(self, p=depends(b1), q=depends(b2), r=depends(c)) -> Item: ...

    _flow(
        a=a, b=b, b1=b1, b2=b2, c=c, j=j,
        edges=(
            edge(START).to(a),
            edge(a).to(b, c),
            edge(b).to(b1, b2),          # nested, joining where the outer one does
            edge(b1, b2, c).to(j),
            edge(j).to(EXIT),
        ),
    )


def test_a_branch_may_fan_out_on_one_arm_only():
    @step
    async def a(self) -> Item: ...

    @step
    async def heavy1(self, x=depends(a)) -> Item: ...

    @step
    async def heavy2(self, x=depends(a)) -> Item: ...

    @step
    async def cheap(self, x=depends(a)) -> Item: ...

    @step
    async def j(
        self,
        p: Item = optional_depends("heavy1"),
        q: Item = optional_depends("heavy2"),
        r: Item = optional_depends("cheap"),
    ) -> Item: ...

    _flow(
        a=a, heavy1=heavy1, heavy2=heavy2, cheap=cheap, j=j,
        edges=(
            edge(START).to(a),
            edge(a).when(lambda r: r.value > 0).to(heavy1, heavy2).otherwise(cheap),
            edge(heavy1, heavy2, cheap).to(j),
            edge(j).to(EXIT),
        ),
    )


def test_an_arm_may_branch_and_loop_internally():
    @step
    async def a(self) -> Item: ...

    @step
    async def b(self, x=depends(a), prev: Item = optional_depends("b_retry")) -> Item: ...

    @step
    async def b_retry(self, x=depends(b)) -> Item: ...

    @step
    async def c(self, x=depends(a)) -> Item: ...

    @step
    async def j(self, p=depends(b), q=depends(c)) -> Item: ...

    _flow(
        a=a, b=b, b_retry=b_retry, c=c, j=j,
        edges=(
            edge(START).to(a),
            edge(a).to(b, c),
            edge(b).when(lambda r: r.value > 0).to(j).otherwise(b_retry),
            edge(b_retry).to(b),
            edge(c).to(j),
            edge(j).to(EXIT),
        ),
    )


def test_the_join_may_loop_back_to_the_node_that_fanned_out():
    @step
    async def a(self, prev: Item = optional_depends("j")) -> Item: ...

    @step
    async def b(self, x=depends(a)) -> Item: ...

    @step
    async def c(self, x=depends(a)) -> Item: ...

    @step
    async def j(self, p=depends(b), q=depends(c)) -> Item: ...

    _flow(
        a=a, b=b, c=c, j=j,
        edges=(
            edge(START).to(a),
            edge(a).to(b, c),
            edge(b, c).to(j),
            edge(j).when(lambda r: r.value > 0).to(EXIT).otherwise(a),
        ),
    )


def test_a_fan_out_may_rejoin_at_the_node_that_fanned_out():
    """`a` is both the source and the join — the loop head fans out every pass."""

    @step
    async def a(self, prev: Item = optional_depends("b")) -> Item: ...

    @step
    async def b(self, x=depends(a)) -> Item: ...

    @step
    async def c(self, x=depends(a)) -> Item: ...

    _flow(
        a=a, b=b, c=c,
        edges=(
            edge(START).to(a),
            edge(a).when(lambda r: r.value > 0).to(EXIT).otherwise(b, c),
            edge(b, c).to(a),
        ),
    )


# --- running ----------------------------------------------------------------------------


class Gate:
    """Proves the arms really are concurrent: each waits for all of them to arrive, so a
    sequential runner deadlocks instead of quietly passing."""

    def __init__(self, parties: int) -> None:
        self.barrier = asyncio.Barrier(parties)
        self.order: list[str] = []

    async def arrive(self, name: str) -> None:
        self.order.append(f"{name}:in")
        await self.barrier.wait()
        self.order.append(f"{name}:out")


GATE: Gate | None = None


class ConcurrentFlow(Flow[Item]):
    @step
    async def seed(self) -> Item:
        return Item(value=1, tag="seed")

    @step
    async def a(self, s=depends(seed)) -> Item:
        assert GATE is not None
        await GATE.arrive("a")
        return Item(value=s.value + 1, tag="a")

    @step
    async def b(self, s=depends(seed)) -> Item:
        assert GATE is not None
        await GATE.arrive("b")
        return Item(value=s.value + 2, tag="b")

    @step
    async def c(self, s=depends(seed)) -> Item:
        assert GATE is not None
        await GATE.arrive("c")
        return Item(value=s.value + 3, tag="c")

    @step
    async def j(self, p=depends(a), q=depends(b), r=depends(c)) -> Item:
        assert GATE is not None
        GATE.order.append("j")
        return Item(value=p.value + q.value + r.value, tag="j")

    edges = (
        edge(START).to(seed),
        edge(seed).to(a, b, c),
        edge(a, b, c).to(j),
        edge(j).to(EXIT),
    )


def test_arms_run_concurrently(run, mem):
    global GATE
    GATE = Gate(3)
    result = run(asyncio.wait_for(ConcurrentFlow().run(persist_service=mem), timeout=5))
    assert result == Item(value=9, tag="j")
    # Every arm entered before any of them left — only possible if all three were in flight.
    assert GATE.order[:3] == ["a:in", "b:in", "c:in"]
    assert GATE.order[-1] == "j"


def test_the_join_runs_once_and_after_every_arm(run, mem):
    global GATE
    GATE = Gate(3)
    run(asyncio.wait_for(ConcurrentFlow().run(persist_service=mem), timeout=5))
    assert GATE.order.count("j") == 1
    assert GATE.order.index("j") > max(GATE.order.index(f"{n}:out") for n in "abc")


def test_every_arm_persists_under_its_own_path(run, mem):
    global GATE
    GATE = Gate(3)
    run(asyncio.wait_for(ConcurrentFlow().run(persist_service=mem, run_id="r1"), timeout=5))
    for name in ("a", "b", "c", "j"):
        assert mem.fetch(f"r1/concurrent/{name}", Item).tag == name


class Boom(Exception):
    pass


class FailingFlow(Flow[Item]):
    cancelled: list[str] = []

    @step
    async def seed(self) -> Item:
        return Item(value=1)

    @step
    async def fails(self, s=depends(seed)) -> Item:
        await asyncio.sleep(0)
        raise Boom("this arm failed")

    @step
    async def slow(self, s=depends(seed)) -> Item:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            FailingFlow.cancelled.append("slow")
            raise
        return Item(value=2)

    @step
    async def j(self, p=depends(fails), q=depends(slow)) -> Item:
        return Item(value=0, tag="reached")

    edges = (
        edge(START).to(seed),
        edge(seed).to(fails, slow),
        edge(fails, slow).to(j),
        edge(j).to(EXIT),
    )


def test_a_failing_arm_propagates_as_itself_and_cancels_its_siblings(run, mem):
    FailingFlow.cancelled.clear()
    with pytest.raises(Boom, match="this arm failed"):
        run(asyncio.wait_for(FailingFlow().run(persist_service=mem), timeout=5))
    assert FailingFlow.cancelled == ["slow"]
    with pytest.raises(FileNotFoundError):
        mem.fetch("failing/j", Item)


class CrashyFlow(Flow[Item]):
    ran: list[str] = []
    crash = True

    @step
    async def seed(self) -> Item:
        CrashyFlow.ran.append("seed")
        return Item(value=1)

    @step
    async def ok(self, s=depends(seed)) -> Item:
        CrashyFlow.ran.append("ok")
        return Item(value=1)

    @step
    async def flaky(self, s=depends(seed)) -> Item:
        CrashyFlow.ran.append("flaky")
        if CrashyFlow.crash:
            CrashyFlow.crash = False
            raise Boom("crash")
        return Item(value=2)

    @step
    async def j(self, p=depends(ok), q=depends(flaky)) -> Item:
        CrashyFlow.ran.append("j")
        return Item(value=p.value + q.value, tag="j")

    edges = (
        edge(START).to(seed),
        edge(seed).to(ok, flaky),
        edge(ok, flaky).to(j),
        edge(j).to(EXIT),
    )


def test_a_crash_inside_a_region_checkpoints_the_node_that_fanned_out(run, mem):
    CrashyFlow.ran.clear()
    CrashyFlow.crash = True
    with pytest.raises(Boom):
        run(CrashyFlow().run(persist_service=mem))
    assert mem.fetch("crashy/_loop_cursor", dict)["next"] == "seed"


def test_resuming_re_runs_the_whole_region(run, mem):
    CrashyFlow.ran.clear()
    CrashyFlow.crash = True
    with pytest.raises(Boom):
        run(CrashyFlow().run(persist_service=mem))
    CrashyFlow.ran.clear()
    assert run(CrashyFlow().run(persist_service=mem)) == Item(value=3, tag="j")
    assert CrashyFlow.ran == ["seed", "ok", "flaky", "j"]


def _budget_flow(max_steps: int):
    """Six nodes, no loop, and only two of them (`seed`, `j`) on the top-level walk — so a
    budget counting only the top-level walk would never trip, and a per-arm budget would
    give each arm its own allowance."""

    @step
    async def seed(self) -> Item:
        return Item(value=1)

    @step
    async def a1(self, s=depends(seed)) -> Item:
        await asyncio.sleep(0)
        return Item(value=1)

    @step
    async def a2(self, x=depends(a1)) -> Item:
        return Item(value=1)

    @step
    async def b1(self, s=depends(seed)) -> Item:
        await asyncio.sleep(0)
        return Item(value=1)

    @step
    async def b2(self, x=depends(b1)) -> Item:
        return Item(value=1)

    @step
    async def j(self, p=depends(a2), q=depends(b2)) -> Item:
        return Item(value=0, tag="reached")

    return type(
        "Budget", (Flow,),
        dict(
            max_steps=max_steps, seed=seed, a1=a1, a2=a2, b1=b1, b2=b2, j=j,
            edges=(
                edge(START).to(seed),
                edge(seed).to(a1, b1),
                edge(a1).to(a2),
                edge(b1).to(b2),
                edge(a2, b2).to(j),
                edge(j).to(EXIT),
            ),
        ),
    )


def test_max_steps_is_one_budget_across_concurrent_arms(run, mem):
    with pytest.raises(RuntimeError, match="ran 5 nodes without reaching EXIT"):
        run(_budget_flow(5)().run(persist_service=mem))
    with pytest.raises(FileNotFoundError):
        mem.fetch("budget/j", Item)


def test_max_steps_exactly_covering_the_run_is_enough(run, mem):
    assert run(_budget_flow(6)().run(persist_service=mem)).tag == "reached"


class Doubler(Flow[Item]):
    src = require(Item)

    @step
    async def double(self, s=depends(src)) -> Item:
        await asyncio.sleep(0)
        return Item(value=s.value * 2, tag="double")

    edges = (edge(START).to(double), edge(double).to(EXIT))


class SubflowArmsFlow(Flow[Item]):
    @step
    async def seed(self) -> Item:
        return Item(value=3)

    left = Doubler.bind(src=seed)
    right = Doubler.bind(src=seed)

    @step
    async def j(self, p=depends(left), q=depends(right)) -> Item:
        return Item(value=p.value + q.value, tag="j")

    edges = (
        edge(START).to(seed),
        edge(seed).to(left, right),
        edge(left, right).to(j),
        edge(j).to(EXIT),
    )


def test_nested_flows_work_as_arms(run, mem):
    assert run(SubflowArmsFlow().run(persist_service=mem, run_id="r")) == Item(value=12, tag="j")
    assert mem.fetch("r/subflowarms/left/double", Item).value == 6
    assert mem.fetch("r/subflowarms/right/double", Item).value == 6
    # Each mount keeps its own cursor, so two arms in flight never share loop state.
    assert mem.fetch("r/subflowarms/left/_loop_cursor", dict)["done"] is True
    assert mem.fetch("r/subflowarms/right/_loop_cursor", dict)["done"] is True


class BranchFanFlow(Flow[Item]):
    seen: list[str] = []

    @step
    async def gate(self, prev: Item = optional_depends("j")) -> Item:
        value = (prev.value + 5) if prev else 0
        BranchFanFlow.seen.append(f"gate:{value}")
        return Item(value=value)

    @step
    async def heavy1(self, g=depends(gate)) -> Item:
        BranchFanFlow.seen.append("heavy1")
        return Item(value=g.value, tag="heavy1")

    @step
    async def heavy2(self, g=depends(gate)) -> Item:
        BranchFanFlow.seen.append("heavy2")
        return Item(value=g.value, tag="heavy2")

    @step
    async def cheap(self, g=depends(gate)) -> Item:
        BranchFanFlow.seen.append("cheap")
        return Item(value=g.value, tag="cheap")

    @step
    async def j(
        self,
        g=depends(gate),
        p: Item = optional_depends("heavy1"),
        q: Item = optional_depends("heavy2"),
        r: Item = optional_depends("cheap"),
    ) -> Item:
        BranchFanFlow.seen.append("j")
        return Item(value=g.value, tag="fanned" if g.value < 5 else "cheap")

    edges = (
        edge(START).to(gate),
        edge(gate).when(lambda i: i.value < 5).to(heavy1, heavy2).otherwise(cheap),
        edge(heavy1, heavy2, cheap).to(j),
        edge(j).when(lambda i: i.value >= 5).to(EXIT).otherwise(gate),
    )


def test_a_branch_fans_out_on_one_arm_and_not_the_other(run, mem):
    BranchFanFlow.seen.clear()
    assert run(BranchFanFlow().run(persist_service=mem)).tag == "cheap"
    # First pass took the fan-out; second took the single node. Same join both times.
    assert BranchFanFlow.seen == [
        "gate:0", "heavy1", "heavy2", "j", "gate:5", "cheap", "j",
    ]


class NestedFanFlow(Flow[Item]):
    """Three levels. `inner` joins strictly inside its arm (at `b_end`); `mid` joins where
    the outermost one does (at `j`)."""

    seen: list[str] = []

    @step
    async def a(self) -> Item:
        NestedFanFlow.seen.append("a")
        return Item(value=1)

    @step
    async def b(self, x=depends(a)) -> Item:
        NestedFanFlow.seen.append("b")
        return Item(value=1)

    @step
    async def b1(self, x=depends(b)) -> Item:
        await asyncio.sleep(0)
        NestedFanFlow.seen.append("b1")
        return Item(value=1)

    @step
    async def b1a(self, x=depends(b1)) -> Item:
        NestedFanFlow.seen.append("b1a")
        return Item(value=1)

    @step
    async def b1b(self, x=depends(b1)) -> Item:
        NestedFanFlow.seen.append("b1b")
        return Item(value=1)

    @step
    async def b_end(self, p=depends(b1a), q=depends(b1b)) -> Item:
        NestedFanFlow.seen.append("b_end")
        return Item(value=p.value + q.value)

    @step
    async def b2(self, x=depends(b)) -> Item:
        NestedFanFlow.seen.append("b2")
        return Item(value=1)

    @step
    async def c(self, x=depends(a)) -> Item:
        NestedFanFlow.seen.append("c")
        return Item(value=1)

    @step
    async def j(self, p=depends(b_end), q=depends(b2), r=depends(c)) -> Item:
        NestedFanFlow.seen.append("j")
        return Item(value=p.value + q.value + r.value, tag="j")

    edges = (
        edge(START).to(a),
        edge(a).to(b, c),                # outer
        edge(b).to(b1, b2),              # nested, joining at the outer join
        edge(b1).to(b1a, b1b),           # nested again, joining strictly inside its arm
        edge(b1a, b1b).to(b_end),
        edge(b_end, b2, c).to(j),
        edge(j).to(EXIT),
    )


def test_nested_fan_outs_run_every_node_exactly_once(run, mem):
    NestedFanFlow.seen.clear()
    assert run(asyncio.wait_for(NestedFanFlow().run(persist_service=mem), timeout=5)) == Item(
        value=4, tag="j"
    )
    assert sorted(NestedFanFlow.seen) == sorted(
        ["a", "b", "b1", "b1a", "b1b", "b_end", "b2", "c", "j"]
    )
    # A join never runs before the arms it closes.
    order = NestedFanFlow.seen
    assert order.index("b_end") > max(order.index("b1a"), order.index("b1b"))
    assert order.index("j") > max(order.index("b_end"), order.index("b2"), order.index("c"))


class LoopHeadFanFlow(Flow[Item]):
    """The join *is* the node that fanned out, so the loop head fans out on every pass."""

    seen: list[str] = []

    @step
    async def head(self, prev: Item = optional_depends("left")) -> Item:
        value = (prev.value + 1) if prev else 0
        LoopHeadFanFlow.seen.append(f"head:{value}")
        return Item(value=value, tag="head")

    @step
    async def left(self, h=depends(head)) -> Item:
        LoopHeadFanFlow.seen.append("left")
        return Item(value=h.value)

    @step
    async def right(self, h=depends(head)) -> Item:
        LoopHeadFanFlow.seen.append("right")
        return Item(value=h.value)

    edges = (
        edge(START).to(head),
        edge(head).when(lambda i: i.value >= 2).to(EXIT).otherwise(left, right),
        edge(left, right).to(head),
    )


def test_a_fan_out_may_rejoin_at_its_own_source_at_run_time(run, mem):
    LoopHeadFanFlow.seen.clear()
    assert run(asyncio.wait_for(LoopHeadFanFlow().run(persist_service=mem), timeout=5)).value == 2
    assert LoopHeadFanFlow.seen == [
        "head:0", "left", "right", "head:1", "left", "right", "head:2",
    ]


class TwoFansFlow(Flow[Item]):
    """Two different fan-outs on two branches of one `.when()` chain."""

    seen: list[str] = []

    @step
    async def pick(self) -> Item:
        return Item(value=1)

    @step
    async def a(self, x=depends(pick)) -> Item:
        TwoFansFlow.seen.append("a")
        return Item(value=1)

    @step
    async def b(self, x=depends(pick)) -> Item:
        TwoFansFlow.seen.append("b")
        return Item(value=1)

    @step
    async def c(self, x=depends(pick)) -> Item:
        TwoFansFlow.seen.append("c")
        return Item(value=1)

    @step
    async def d(self, x=depends(pick)) -> Item:
        TwoFansFlow.seen.append("d")
        return Item(value=1)

    @step
    async def e(self, x=depends(pick)) -> Item:
        TwoFansFlow.seen.append("e")
        return Item(value=1)

    @step
    async def j(
        self,
        p: Item = optional_depends("a"),
        q: Item = optional_depends("c"),
        r: Item = optional_depends("e"),
    ) -> Item:
        TwoFansFlow.seen.append("j")
        return Item(value=0, tag="j")

    edges = (
        edge(START).to(pick),
        edge(pick).when(lambda i: i.value == 0).to(a, b).when(lambda i: i.value == 1).to(c, d).otherwise(e),
        edge(a, b, c, d, e).to(j),
        edge(j).to(EXIT),
    )


def test_two_fan_outs_on_one_branch_chain_route_independently(run, mem):
    TwoFansFlow.seen.clear()
    run(asyncio.wait_for(TwoFansFlow().run(persist_service=mem), timeout=5))
    assert TwoFansFlow.seen == ["c", "d", "j"]      # `pick` returns 1, so the second branch


class Splitter(Flow[Item]):
    """A child flow that itself fans out — mounted as an arm of its parent's fan-out."""

    src = require(Item)

    @step
    async def head(self, s=depends(src)) -> Item:
        return Item(value=s.value)

    @step
    async def x(self, h=depends(head)) -> Item:
        await asyncio.sleep(0)
        return Item(value=h.value * 2)

    @step
    async def y(self, h=depends(head)) -> Item:
        await asyncio.sleep(0)
        return Item(value=h.value * 3)

    @step
    async def sum_(self, p=depends(x), q=depends(y)) -> Item:
        return Item(value=p.value + q.value, tag="sum")

    edges = (
        edge(START).to(head),
        edge(head).to(x, y),
        edge(x, y).to(sum_),
        edge(sum_).to(EXIT),
    )


class NestedSubflowFanFlow(Flow[Item]):
    @step
    async def seed(self) -> Item:
        return Item(value=1)

    left = Splitter.bind(src=seed)
    right = Splitter.bind(src=seed)

    @step
    async def noted(self, s=depends(seed)):
        """No return annotation: persists nothing, so it is pure ordering."""

    @step
    async def j(self, p=depends(left), q=depends(right), n=depends(noted)) -> Item:
        assert n is None
        return Item(value=p.value + q.value, tag="j")

    edges = (
        edge(START).to(seed),
        edge(seed).to(left, right, noted),
        edge(left, right, noted).to(j),
        edge(j).to(EXIT),
    )


def test_a_child_flow_that_fans_out_works_as_an_arm(run, mem):
    result = run(asyncio.wait_for(NestedSubflowFanFlow().run(persist_service=mem, run_id="r"), timeout=5))
    assert result == Item(value=10, tag="j")
    for mount in ("left", "right"):
        assert mem.fetch(f"r/nestedsubflowfan/{mount}/x", Item).value == 2
        assert mem.fetch(f"r/nestedsubflowfan/{mount}/y", Item).value == 3
        assert mem.fetch(f"r/nestedsubflowfan/{mount}", Item).value == 5
    # A step with no return annotation is an arm like any other, and writes nothing.
    with pytest.raises(FileNotFoundError):
        mem.fetch("r/nestedsubflowfan/noted", Item)


class EnterArmFlow(Flow[Item]):
    ran: list[str] = []
    crash = False

    @step
    async def seed(self) -> Item:
        EnterArmFlow.ran.append("seed")
        return Item(value=1)

    @step
    async def a1(self, s=depends(seed)) -> Item:
        EnterArmFlow.ran.append("a1")
        if EnterArmFlow.crash:
            raise Boom("crash inside the arm")
        return Item(value=10)

    @step
    async def b1(self, s=depends(seed)) -> Item:
        EnterArmFlow.ran.append("b1")
        return Item(value=100)

    @step
    async def j(self, p=depends(a1), q=depends(b1)) -> Item:
        EnterArmFlow.ran.append("j")
        return Item(value=p.value + q.value, tag="j")

    edges = (
        edge(START).to(seed),
        edge(seed).to(a1, b1),
        edge(a1, b1).to(j),
        edge(j).to(EXIT),
    )


def test_a_cursor_left_inside_an_arm_is_not_resumed_from(run, mem):
    """`follow_edges` into an arm is an explicit one-armed pass, and it checkpoints. A plain
    `run()` afterwards must not silently continue that pass — the siblings never started."""
    EnterArmFlow.crash = False
    run(EnterArmFlow().run(persist_service=mem))          # everything on the backend

    EnterArmFlow.crash = True
    EnterArmFlow.ran.clear()
    with pytest.raises(Boom):
        run(EnterArmFlow().run("a1", follow_edges=True, persist_service=mem))
    assert EnterArmFlow.ran == ["a1"]                     # the one-armed pass, as asked
    assert mem.fetch("enterarm/_loop_cursor", dict)["next"] == "a1"

    EnterArmFlow.crash = False
    EnterArmFlow.ran.clear()
    assert run(EnterArmFlow().run(persist_service=mem)) == Item(value=110, tag="j")
    assert EnterArmFlow.ran == ["seed", "a1", "b1", "j"]      # from START, not from `a1`
