"""The edge builder and the graph it validates. Everything here raises while the class is
being created — a malformed graph never reaches a run."""

import pytest
from pydantic import BaseModel

from stepper import EXIT, START, Flow, depends, edge, optional_depends, step


class Item(BaseModel):
    value: int


def _flow(**body):
    """Build a Flow subclass from a dict body, so a bad graph raises here rather than at
    import time."""
    return type("Made", (Flow,), body)


# --- the builder ----------------------------------------------------------------------


def test_when_without_to_is_rejected():
    e = edge(START).when(lambda r: True)
    with pytest.raises(TypeError, match=r"\.when\(\.\.\.\) with no \.to"):
        e.check()


def test_two_whens_in_a_row_are_rejected():
    e = edge(START).when(lambda r: True)
    with pytest.raises(TypeError, match="twice with no"):
        e.when(lambda r: False)


def test_when_after_an_unconditional_to_is_rejected():
    e = edge(START).to(EXIT)
    with pytest.raises(TypeError, match="after an unconditional"):
        e.when(lambda r: True)


def test_otherwise_without_a_when_is_rejected():
    with pytest.raises(TypeError, match="with no .when"):
        edge(START).otherwise(EXIT)


def test_otherwise_twice_is_rejected():
    e = edge(START).when(lambda r: True).to(EXIT).otherwise(EXIT)
    with pytest.raises(TypeError, match="declared twice"):
        e.otherwise(EXIT)


def test_when_after_otherwise_is_rejected():
    e = edge(START).when(lambda r: True).to(EXIT).otherwise(EXIT)
    with pytest.raises(TypeError, match="after .otherwise"):
        e.when(lambda r: False)


def test_otherwise_on_an_unconditional_edge_is_rejected():
    e = edge(START).to(EXIT)
    with pytest.raises(TypeError, match="on an unconditional edge"):
        e.otherwise(EXIT)


def test_an_unconditional_to_after_a_branch_is_rejected():
    e = edge(START).when(lambda r: True).to(EXIT)
    with pytest.raises(TypeError, match="mixed with branches"):
        e.to(EXIT)


def test_an_edge_that_routes_nowhere_is_rejected():
    with pytest.raises(TypeError, match="routes nowhere"):
        edge(START).check()


def test_a_branch_without_otherwise_is_rejected():
    e = edge(START).when(lambda r: True).to(EXIT)
    with pytest.raises(TypeError, match="needs .otherwise"):
        e.check()


def test_first_matching_predicate_wins():
    a, b, c = object(), object(), object()
    e = edge(START).when(lambda r: r > 1).to(a).when(lambda r: r > 0).to(b).otherwise(c)
    assert e.resolve(2) is a
    assert e.resolve(1) is b
    assert e.resolve(-1) is c


# --- the graph ------------------------------------------------------------------------


@step
async def _orphan() -> Item:
    return Item(value=1)


def _one_step():
    @step
    async def only(self) -> Item:
        return Item(value=1)

    return only


def test_a_graph_needs_an_entry():
    only = _one_step()
    with pytest.raises(TypeError, match="no edge\\(START\\)"):
        _flow(only=only, edges=(edge(only).to(EXIT),))


def test_two_entries_are_rejected():
    only = _one_step()
    with pytest.raises(TypeError, match="more than one edge\\(START\\)"):
        _flow(only=only, edges=(edge(START).to(only), edge(START).to(only), edge(only).to(EXIT)))


def test_a_branching_entry_is_rejected():
    only = _one_step()
    with pytest.raises(TypeError, match="edge\\(START\\) must be unconditional"):
        _flow(
            only=only,
            edges=(edge(START).when(lambda r: True).to(only).otherwise(only), edge(only).to(EXIT)),
        )


def test_start_straight_to_exit_is_rejected():
    only = _one_step()
    with pytest.raises(TypeError, match="runs nothing"):
        _flow(only=only, edges=(edge(START).to(EXIT), edge(only).to(EXIT)))


def test_two_edges_from_one_node_are_rejected():
    only = _one_step()
    with pytest.raises(TypeError, match="more than one edge"):
        _flow(only=only, edges=(edge(START).to(only), edge(only).to(EXIT), edge(only).to(EXIT)))


def test_routing_to_a_node_of_another_flow_is_rejected():
    only = _one_step()
    with pytest.raises(TypeError, match="not a node of this flow"):
        _flow(only=only, edges=(edge(START).to(only), edge(only).to(_orphan)))


def test_an_edge_out_of_a_node_of_another_flow_is_rejected():
    only = _one_step()
    with pytest.raises(TypeError, match="not a node of this flow"):
        _flow(only=only, edges=(edge(START).to(only), edge(only).to(EXIT), edge(_orphan).to(EXIT)))


def test_a_node_with_no_edge_out_is_rejected():
    a, b = _one_step(), _one_step()
    with pytest.raises(TypeError, match="declare no edge"):
        _flow(only=a, second=b, edges=(edge(START).to(a), edge(a).to(EXIT)))


def test_an_unreachable_node_is_rejected():
    a, b = _one_step(), _one_step()
    with pytest.raises(TypeError, match="unreachable from START"):
        _flow(only=a, second=b, edges=(edge(START).to(a), edge(a).to(EXIT), edge(b).to(EXIT)))


def test_a_node_with_no_route_to_exit_is_rejected():
    with pytest.raises(TypeError, match="no route to EXIT"):

        class Stuck(Flow[Item]):
            @step
            async def a(self) -> Item:
                return Item(value=1)

            @step
            async def b(self, prev=depends(a)) -> Item:
                return prev

            # a -> b -> b forever; nothing reaches EXIT.
            edges = (edge(START).to(a), edge(a).to(b), edge(b).to(b))


def test_a_branch_arm_that_cannot_finish_is_rejected():
    """Per-node, not "there is an EXIT somewhere" — one arm dropping into a closed cycle
    is exactly as broken as a graph with no EXIT at all."""
    with pytest.raises(TypeError, match="no route to EXIT"):

        class Trapped(Flow[Item]):
            @step
            async def a(self) -> Item:
                return Item(value=1)

            @step
            async def b(self, prev=depends(a)) -> Item:
                return prev

            @step
            async def c(self, prev=depends(a)) -> Item:
                return prev

            edges = (
                edge(START).to(a),
                edge(a).when(lambda i: i.value > 0).to(EXIT).otherwise(b),
                edge(b).to(c),
                edge(c).to(b),
            )


# --- depends() checked against the graph ----------------------------------------------


def test_a_required_dep_must_be_guaranteed_on_every_path():
    with pytest.raises(TypeError, match="not guaranteed to have run before it"):

        class TooSoon(Flow[Item]):
            @step
            async def consumer(self, prev: Item = depends("producer")) -> Item:
                return prev

            @step
            async def producer(self) -> Item:
                return Item(value=1)

            edges = (edge(START).to(consumer), edge(consumer).to(producer), edge(producer).to(EXIT))


def test_an_optional_dep_needs_a_path_that_could_supply_it():
    with pytest.raises(TypeError, match="no path leads from"):

        class Dead(Flow[Item]):
            @step
            async def consumer(self, prev: Item | None = optional_depends("producer")) -> Item:
                return prev or Item(value=0)

            @step
            async def producer(self) -> Item:
                return Item(value=1)

            # producer runs after consumer and then exits, so it can never precede it.
            edges = (edge(START).to(consumer), edge(consumer).to(producer), edge(producer).to(EXIT))


def test_a_back_edge_makes_an_optional_dep_legal():
    class Loop(Flow[Item]):
        @step
        async def edit(self, prev: Item | None = optional_depends("audit")) -> Item:
            return Item(value=(prev.value if prev else 0) + 1)

        @step
        async def audit(self, draft=depends(edit)) -> Item:
            return draft

        edges = (
            edge(START).to(edit),
            edge(edit).to(audit),
            edge(audit).when(lambda i: i.value >= 3).to(EXIT).otherwise(edit),
        )

    assert Loop.plan().graph.terminals() == ["audit"]
