"""Graph validation: every rejected shape, and every accepted one.

Two tables. `INVALID` pairs a graph with the message it must raise at class creation;
`VALID` holds shapes that must build cleanly. The point of the second table is the half
that's easy to forget — a check that rejects a legal loop is as broken as one that lets a
dead graph through, and the back-edge shapes below are exactly the ones a naive "producer
must come first" rule would wrongly reject.

Each case is a thunk that declares a `Stage` subclass, since validation runs during class
creation. Steps are created fresh per call (a `Step` can only be claimed by one stage).

No ``from __future__ import annotations`` — these stages are declared inside functions.
"""

import pytest
from pydantic import BaseModel

from stepper import EXIT, START, Stage, depends, edge, optional_depends, step


class V(BaseModel):
    n: int


def _v() -> V:
    return V(n=0)


# --- rejected shapes --------------------------------------------------------------


def no_start():
    class S(Stage):
        @step
        async def a(self) -> V:
            return _v()

        steps = (a,)
        edges = (edge(a).to(EXIT),)


def two_starts():
    class S(Stage):
        @step
        async def a(self) -> V:
            return _v()

        @step
        async def b(self) -> V:
            return _v()

        steps = (a, b)
        edges = (edge(START).to(a), edge(START).to(b), edge(a).to(EXIT), edge(b).to(EXIT))


def branching_start():
    class S(Stage):
        @step
        async def a(self) -> V:
            return _v()

        steps = (a,)
        edges = (edge(START).when(lambda r: True).to(a).otherwise(a), edge(a).to(EXIT))


def start_straight_to_exit():
    class S(Stage):
        @step
        async def a(self) -> V:
            return _v()

        steps = (a,)
        edges = (edge(START).to(EXIT), edge(a).to(EXIT))


def step_with_no_edge_out():
    class S(Stage):
        @step
        async def a(self) -> V:
            return _v()

        @step
        async def b(self) -> V:
            return _v()

        steps = (a, b)
        edges = (edge(START).to(a), edge(a).to(EXIT))


def branch_without_otherwise():
    class S(Stage):
        @step
        async def a(self) -> V:
            return _v()

        steps = (a,)
        edges = (edge(START).to(a), edge(a).when(lambda r: True).to(EXIT))


def when_without_to():
    class S(Stage):
        @step
        async def a(self) -> V:
            return _v()

        steps = (a,)
        edges = (edge(START).to(a), edge(a).when(lambda r: True))


def otherwise_without_when():
    edge(_solo()).otherwise(EXIT)


def otherwise_twice():
    edge(_solo()).when(lambda r: True).to(EXIT).otherwise(EXIT).otherwise(EXIT)


def when_after_unconditional():
    edge(_solo()).to(EXIT).when(lambda r: True)


def unconditional_after_branch():
    edge(_solo()).when(lambda r: True).to(EXIT).to(EXIT)


def when_after_otherwise():
    edge(_solo()).when(lambda r: True).to(EXIT).otherwise(EXIT).when(lambda r: True)


def when_twice_with_no_to():
    edge(_solo()).when(lambda r: True).when(lambda r: True)


def two_edges_for_one_step():
    class S(Stage):
        @step
        async def a(self) -> V:
            return _v()

        steps = (a,)
        edges = (edge(START).to(a), edge(a).to(EXIT), edge(a).to(EXIT))


def target_not_a_listed_step():
    class Other(Stage):
        @step
        async def stray(self) -> V:
            return _v()

        steps = (stray,)

    class S(Stage):
        @step
        async def a(self) -> V:
            return _v()

        steps = (a,)
        edges = (edge(START).to(a), edge(a).when(lambda r: True).to(Other.stray).otherwise(EXIT))


def source_not_a_listed_step():
    class Other(Stage):
        @step
        async def stray(self) -> V:
            return _v()

        steps = (stray,)

    class S(Stage):
        @step
        async def a(self) -> V:
            return _v()

        steps = (a,)
        edges = (edge(START).to(a), edge(a).to(EXIT), edge(Other.stray).to(EXIT))


def unreachable_step():
    class S(Stage):
        @step
        async def a(self) -> V:
            return _v()

        @step
        async def orphan(self) -> V:
            return _v()

        steps = (a, orphan)
        edges = (edge(START).to(a), edge(a).to(EXIT), edge(orphan).to(EXIT))


def unreachable_island():
    """Two steps wired to each other but never entered from START."""
    class S(Stage):
        @step
        async def a(self) -> V:
            return _v()

        @step
        async def x(self) -> V:
            return _v()

        @step
        async def y(self) -> V:
            return _v()

        steps = (a, x, y)
        edges = (edge(START).to(a), edge(a).to(EXIT), edge(x).to(y), edge(y).to(EXIT))


def no_route_to_exit():
    class S(Stage):
        @step
        async def a(self) -> V:
            return _v()

        steps = (a,)
        edges = (edge(START).to(a), edge(a).to(a))


def no_route_to_exit_from_a_bigger_cycle():
    class S(Stage):
        @step
        async def a(self) -> V:
            return _v()

        @step
        async def b(self) -> V:
            return _v()

        @step
        async def c(self) -> V:
            return _v()

        steps = (a, b, c)
        edges = (edge(START).to(a), edge(a).to(b), edge(b).to(c), edge(c).to(a))


def required_dep_never_runs_first():
    """START -> b, but b needs a."""
    class S(Stage):
        @step
        async def a(self) -> V:
            return _v()

        @step
        async def b(self, x=depends(a)) -> V:
            return x

        steps = (a, b)
        edges = (edge(START).to(b), edge(b).to(a), edge(a).when(lambda r: True).to(EXIT).otherwise(b))


def required_dep_only_on_one_branch():
    """`join` is reachable both through `mid` and around it."""
    class S(Stage):
        @step
        async def head(self) -> V:
            return _v()

        @step
        async def mid(self, x=depends(head)) -> V:
            return x

        @step
        async def join(self, x=depends(mid)) -> V:
            return x

        steps = (head, mid, join)
        edges = (
            edge(START).to(head),
            edge(head).when(lambda r: True).to(mid).otherwise(join),
            edge(mid).to(join),
            edge(join).to(EXIT),
        )


def required_dep_behind_a_loop_that_can_skip_it():
    """`b` is only visited on some passes, so `c` can't require it."""
    class S(Stage):
        @step
        async def a(self) -> V:
            return _v()

        @step
        async def b(self) -> V:
            return _v()

        @step
        async def c(self, x=depends(b)) -> V:
            return x

        steps = (a, b, c)
        edges = (
            edge(START).to(a),
            edge(a).when(lambda r: True).to(b).otherwise(c),
            edge(b).to(c),
            edge(c).when(lambda r: True).to(EXIT).otherwise(a),
        )


def optional_dep_that_can_never_be_satisfied():
    """`a` runs after `b` but never routes back, so `b`'s parameter is None forever."""
    class S(Stage):
        @step
        async def a(self) -> V:
            return _v()

        @step
        async def b(self, x=optional_depends(a)) -> V:
            return _v()

        steps = (a, b)
        edges = (edge(START).to(b), edge(b).to(a), edge(a).to(EXIT))


def optional_dep_on_a_parallel_terminal_branch():
    """`left` and `right` are alternatives — neither ever precedes the other."""
    class S(Stage):
        @step
        async def head(self) -> V:
            return _v()

        @step
        async def left(self) -> V:
            return _v()

        @step
        async def right(self, x=optional_depends(left)) -> V:
            return _v()

        steps = (head, left, right)
        edges = (
            edge(START).to(head),
            edge(head).when(lambda r: True).to(left).otherwise(right),
            edge(left).to(EXIT),
            edge(right).to(EXIT),
        )


def unknown_back_edge_name():
    class S(Stage):
        @step
        async def a(self, prev=optional_depends("ghost")) -> V:
            return _v()

        steps = (a,)


def trapped_branch_arm():
    """One arm exits, the other drops into a sub-cycle it can never leave. `EXIT` is
    reachable *somewhere*, which is not the same as reachable from everywhere."""
    class S(Stage):
        @step
        async def a(self) -> V:
            return _v()

        @step
        async def b(self) -> V:
            return _v()

        @step
        async def c(self) -> V:
            return _v()

        steps = (a, b, c)
        edges = (
            edge(START).to(a),
            edge(a).when(lambda r: True).to(EXIT).otherwise(b),
            edge(b).to(c),
            edge(c).to(b),
        )


def required_self_dependency():
    class S(Stage):
        @step
        async def a(self, prev=depends("a")) -> V:
            return prev

        steps = (a,)
        edges = (edge(START).to(a), edge(a).when(lambda r: True).to(EXIT).otherwise(a))


def required_dep_only_from_the_second_pass():
    """`b` requires `c`, which only exists once the loop has been round once — so the
    very first visit to `b` has nothing to fetch."""
    class S(Stage):
        @step
        async def a(self) -> V:
            return _v()

        @step
        async def b(self, x=depends("c")) -> V:
            return x

        @step
        async def c(self) -> V:
            return _v()

        steps = (a, b, c)
        edges = (
            edge(START).to(a),
            edge(a).to(b),
            edge(b).to(c),
            edge(c).when(lambda r: True).to(EXIT).otherwise(b),
        )


def edge_from_a_same_named_step_on_another_stage():
    """Both stages own a step called `a`. Naming the wrong one must be rejected, not
    silently rebound to the local step of the same name."""
    class Other(Stage):
        @step
        async def a(self) -> V:
            return _v()

        steps = (a,)

    class S(Stage):
        @step
        async def a(self) -> V:
            return _v()

        @step
        async def z(self) -> V:
            return _v()

        steps = (a, z)
        edges = (edge(START).to(a), edge(Other.a).to(z), edge(z).to(EXIT))


def otherwise_on_an_unconditional_edge():
    edge(_solo()).to(EXIT).otherwise(EXIT)


def two_unconditional_targets():
    edge(_solo()).to(EXIT).to(EXIT)


def edge_declared_with_no_target():
    class S(Stage):
        @step
        async def a(self) -> V:
            return _v()

        steps = (a,)
        edges = (edge(START).to(a), edge(a))


def max_steps_below_one():
    class S(Stage):
        @step
        async def a(self) -> V:
            return _v()

        steps = (a,)
        edges = (edge(START).to(a), edge(a).to(EXIT))
        max_steps = 0


def step_named_like_the_cursor():
    class S(Stage):
        @step
        async def _loop_cursor(self) -> V:
            return _v()

        steps = (_loop_cursor,)


INVALID = [
    (no_start, "nothing says where the graph begins"),
    (two_starts, r"more than one edge\(START\)"),
    (branching_start, r"edge\(START\) must be unconditional"),
    (start_straight_to_exit, r"edge\(START\).to\(EXIT\) runs nothing"),
    (step_with_no_edge_out, "declare no edge"),
    (branch_without_otherwise, r"needs \.otherwise"),
    (when_without_to, r"\.when\(\.\.\.\) with no \.to"),
    (otherwise_without_when, r"\.otherwise\(\.\.\.\) with no \.when"),
    (otherwise_twice, r"\.otherwise\(\.\.\.\) declared twice"),
    (when_after_unconditional, "after an unconditional"),
    (unconditional_after_branch, "mixed with branches"),
    (when_after_otherwise, r"\.when\(\.\.\.\) after \.otherwise"),
    (when_twice_with_no_to, r"\.when\(\.\.\.\) twice"),
    (two_edges_for_one_step, "has more than one edge"),
    (target_not_a_listed_step, "routes to 'stray'"),
    (source_not_a_listed_step, "'stray' is not in steps"),
    (edge_from_a_same_named_step_on_another_stage, "'a' is not in steps"),
    (unreachable_step, "orphan is unreachable from START"),
    (unreachable_island, "x, y is unreachable from START"),
    (no_route_to_exit, "a has no route to EXIT"),
    (no_route_to_exit_from_a_bigger_cycle, "a, b, c has no route to EXIT"),
    (trapped_branch_arm, "b, c has no route to EXIT"),
    (required_dep_never_runs_first, "'b' requires 'a'.*not guaranteed to have run"),
    (required_dep_only_on_one_branch, "'join' requires 'mid'.*not guaranteed"),
    (required_dep_behind_a_loop_that_can_skip_it, "'c' requires 'b'.*not guaranteed"),
    (required_self_dependency, "'a' requires 'a'.*not guaranteed"),
    (required_dep_only_from_the_second_pass, "'b' requires 'c'.*not guaranteed"),
    (optional_dep_that_can_never_be_satisfied, "no path leads from 'a' back to 'b'"),
    (optional_dep_on_a_parallel_terminal_branch, "no path leads from 'left' back to 'right'"),
    (unknown_back_edge_name, "which is not a step on"),
    (otherwise_on_an_unconditional_edge, r"\.otherwise\(\.\.\.\) on an unconditional edge"),
    (two_unconditional_targets, "mixed with branches"),
    (edge_declared_with_no_target, "routes nowhere"),
    (max_steps_below_one, "max_steps must be at least 1"),
    (step_named_like_the_cursor, "reserved for the loop cursor"),
]


@pytest.mark.parametrize("build,message", INVALID, ids=lambda v: getattr(v, "__name__", ""))
def test_invalid_graph_is_rejected(build, message):
    with pytest.raises(TypeError, match=message):
        build()


# --- accepted shapes --------------------------------------------------------------


def straight_line():
    class S(Stage):
        @step
        async def a(self) -> V:
            return _v()

        @step
        async def b(self, x=depends(a)) -> V:
            return x

        steps = (a, b)
        edges = (edge(START).to(a), edge(a).to(b), edge(b).to(EXIT))

    return S


def self_loop_with_exit():
    class S(Stage):
        @step
        async def a(self) -> V:
            return _v()

        steps = (a,)
        edges = (edge(START).to(a), edge(a).when(lambda r: r.n >= 1).to(EXIT).otherwise(a))

    return S


def two_step_cycle_with_back_edge_dep():
    """The core loop shape: `a` reads the previous pass's `b`."""
    class S(Stage):
        @step
        async def a(self, prev=optional_depends("b")) -> V:
            return _v()

        @step
        async def b(self, x=depends(a)) -> V:
            return x

        steps = (a, b)
        edges = (edge(START).to(a), edge(a).to(b), edge(b).when(lambda r: r.n >= 1).to(EXIT).otherwise(a))

    return S


def entry_reads_a_later_step_that_loops_back():
    """The case that a naive 'producer must come first' rule would wrongly reject: the
    graph starts at `b`, and `a` only ever runs after it — but it does route back."""
    class S(Stage):
        @step
        async def a(self) -> V:
            return _v()

        @step
        async def b(self, prev=optional_depends(a)) -> V:
            return _v()

        steps = (a, b)
        edges = (edge(START).to(b), edge(b).to(a), edge(a).when(lambda r: r.n >= 1).to(EXIT).otherwise(b))

    return S


def branch_to_two_terminals():
    class S(Stage):
        @step
        async def head(self) -> V:
            return _v()

        @step
        async def left(self, x=depends(head)) -> V:
            return x

        @step
        async def right(self, x=depends(head)) -> V:
            return x

        steps = (head, left, right)
        edges = (
            edge(START).to(head),
            edge(head).when(lambda r: True).to(left).otherwise(right),
            edge(left).to(EXIT),
            edge(right).to(EXIT),
        )

    return S


def join_whose_dep_is_on_every_path():
    class S(Stage):
        @step
        async def head(self) -> V:
            return _v()

        @step
        async def left(self, x=depends(head)) -> V:
            return x

        @step
        async def right(self, x=depends(head)) -> V:
            return x

        @step
        async def join(self, x=depends(head)) -> V:
            return x

        steps = (head, left, right, join)
        edges = (
            edge(START).to(head),
            edge(head).when(lambda r: True).to(left).otherwise(right),
            edge(left).to(join),
            edge(right).to(join),
            edge(join).to(EXIT),
        )

    return S


def two_cycles_sharing_a_node():
    class S(Stage):
        @step
        async def hub(self, prev=optional_depends("inner")) -> V:
            return _v()

        @step
        async def inner(self, x=depends(hub)) -> V:
            return x

        @step
        async def outer(self, x=depends(hub)) -> V:
            return x

        steps = (hub, inner, outer)
        edges = (
            edge(START).to(hub),
            edge(hub).when(lambda r: True).to(inner).otherwise(outer),
            edge(inner).when(lambda r: r.n >= 1).to(EXIT).otherwise(hub),
            edge(outer).to(hub),
        )

    return S


def chain_with_a_loop_in_the_middle():
    class S(Stage):
        @step
        async def setup(self) -> V:
            return _v()

        @step
        async def body(self, base=depends(setup), prev=optional_depends("echo")) -> V:
            return base

        @step
        async def echo(self, x=depends(body)) -> V:
            return x

        @step
        async def report(self, x=depends(body)) -> V:
            return x

        steps = (setup, body, echo, report)
        edges = (
            edge(START).to(setup),
            edge(setup).to(body),
            edge(body).to(echo),
            edge(echo).when(lambda r: r.n >= 1).to(report).otherwise(body),
            edge(report).to(EXIT),
        )

    return S


def cross_stage_dep_is_not_the_graphs_business():
    """A `depends()` on another stage is a disk input, so the graph says nothing about it."""
    class Upstream(Stage):
        @step
        async def prod(self) -> V:
            return _v()

        steps = (prod,)

    class S(Stage):
        @step
        async def a(self, x=depends(Upstream.prod)) -> V:
            return x

        steps = (a,)
        edges = (edge(START).to(a), edge(a).to(EXIT))

    return S


def multiple_deps_all_guaranteed():
    class S(Stage):
        @step
        async def a(self) -> V:
            return _v()

        @step
        async def b(self, x=depends(a)) -> V:
            return x

        @step
        async def c(self, x=depends(a), y=depends(b)) -> V:
            return y

        steps = (a, b, c)
        edges = (edge(START).to(a), edge(a).to(b), edge(b).to(c), edge(c).to(EXIT))

    return S


def nested_cycles():
    """An inner loop (`body <-> inner`) sitting inside an outer one (`tail -> head`)."""
    class S(Stage):
        @step
        async def head(self, prev=optional_depends("tail")) -> V:
            return _v()

        @step
        async def body(self, x=depends(head)) -> V:
            return x

        @step
        async def inner(self, x=depends(body)) -> V:
            return x

        @step
        async def tail(self, x=depends(body)) -> V:
            return x

        steps = (head, body, inner, tail)
        edges = (
            edge(START).to(head),
            edge(head).to(body),
            edge(body).when(lambda r: r.n >= 1).to(tail).otherwise(inner),
            edge(inner).to(body),
            edge(tail).when(lambda r: r.n >= 2).to(EXIT).otherwise(head),
        )

    return S


def two_back_edges_into_one_head():
    class S(Stage):
        @step
        async def head(self, prev=optional_depends("left")) -> V:
            return _v()

        @step
        async def left(self, x=depends(head)) -> V:
            return x

        @step
        async def right(self, x=depends(head)) -> V:
            return x

        steps = (head, left, right)
        edges = (
            edge(START).to(head),
            edge(head).when(lambda r: True).to(left).otherwise(right),
            edge(left).when(lambda r: r.n >= 1).to(EXIT).otherwise(head),
            edge(right).to(head),
        )

    return S


def diamond_with_a_cycle():
    """`join` is both the meeting point of the branches and the loop's back edge."""
    class S(Stage):
        @step
        async def head(self, prev=optional_depends("join")) -> V:
            return _v()

        @step
        async def left(self, x=depends(head)) -> V:
            return x

        @step
        async def right(self, x=depends(head)) -> V:
            return x

        @step
        async def join(self, x=depends(head)) -> V:
            return x

        steps = (head, left, right, join)
        edges = (
            edge(START).to(head),
            edge(head).when(lambda r: True).to(left).otherwise(right),
            edge(left).to(join),
            edge(right).to(join),
            edge(join).when(lambda r: r.n >= 1).to(EXIT).otherwise(head),
        )

    return S


def optional_self_dependency_on_a_self_loop():
    """`a` reads its own previous pass — legal because `a` routes back to itself."""
    class S(Stage):
        @step
        async def a(self, prev=optional_depends("a")) -> V:
            return _v()

        steps = (a,)
        edges = (edge(START).to(a), edge(a).when(lambda r: r.n >= 1).to(EXIT).otherwise(a))

    return S


def cross_stage_optional_dep():
    class Upstream(Stage):
        @step
        async def prod(self) -> V:
            return _v()

        steps = (prod,)

    class S(Stage):
        @step
        async def a(self, x=optional_depends(Upstream.prod)) -> V:
            return _v()

        steps = (a,)
        edges = (edge(START).to(a), edge(a).to(EXIT))

    return S


VALID = [
    straight_line,
    nested_cycles,
    two_back_edges_into_one_head,
    diamond_with_a_cycle,
    optional_self_dependency_on_a_self_loop,
    cross_stage_optional_dep,
    self_loop_with_exit,
    two_step_cycle_with_back_edge_dep,
    entry_reads_a_later_step_that_loops_back,
    branch_to_two_terminals,
    join_whose_dep_is_on_every_path,
    two_cycles_sharing_a_node,
    chain_with_a_loop_in_the_middle,
    cross_stage_dep_is_not_the_graphs_business,
    multiple_deps_all_guaranteed,
]


@pytest.mark.parametrize("build", VALID, ids=lambda v: v.__name__)
def test_valid_graph_is_accepted(build):
    """Not just "didn't raise" — the stage must actually come out edge-driven. Asserting
    only the absence of an exception would still pass if `build_graph` returned None and
    the stage silently fell back to the DAG scheduler."""
    S = build()
    assert S._graph is not None, "declared edges but did not build a graph"
    assert S._scheduler is None, "edge-driven stage must not also build a DAG scheduler"
    assert S._graph.entry.name in {s.name for s in S.steps}
    # Every step has an edge out, so every step is routable.
    for s in S.steps:
        assert S._graph.edge_for(s.name) is not None


def _solo():
    """An unclaimed step, for the builder-shape cases that raise before any stage exists."""

    @step
    async def solo() -> V:
        return _v()

    return solo
