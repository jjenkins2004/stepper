"""Everything a flow declares is about itself, so everything is checked when the class is
created. These tests are the catalogue of what that catches — each one asserts the failure
happens at `class ...:` rather than at run time.
"""

import pytest
from pydantic import BaseModel

from stepper import EXIT, START, Flow, depends, edge, require, step


class Item(BaseModel):
    value: int


class Other(BaseModel):
    text: str


class Source(Flow[Item]):
    seed = require(Item)

    @step
    async def emit(self, item=depends(seed)) -> Item:
        return item

    edges = (edge(START).to(emit), edge(emit).to(EXIT))


class Closed(Flow[Item]):
    @step
    async def only(self) -> Item:
        return Item(value=1)

    edges = (edge(START).to(only), edge(only).to(EXIT))


# --- wiring stays inside the flow -----------------------------------------------------


def test_depends_on_another_flows_step_is_rejected():
    with pytest.raises(TypeError, match="belongs to another flow"):

        class Reaching(Flow[Item]):
            @step
            async def take(self, item=depends(Closed.only)) -> Item:
                return item

            edges = (edge(START).to(take), edge(take).to(EXIT))


def test_depends_by_name_must_name_something_declared_here():
    with pytest.raises(TypeError, match="not declared on"):

        class Misnamed(Flow[Item]):
            @step
            async def take(self, item: Item = depends("nope")) -> Item:
                return item

            edges = (edge(START).to(take), edge(take).to(EXIT))


def test_unwired_parameter_is_rejected():
    with pytest.raises(TypeError, match="must be wired with depends"):

        class Bare(Flow[Item]):
            @step
            async def take(self, item) -> Item:  # type: ignore[no-untyped-def]
                return item

            edges = (edge(START).to(take), edge(take).to(EXIT))


# --- bindings -------------------------------------------------------------------------


def test_missing_binding_is_rejected():
    with pytest.raises(TypeError, match="requires seed"):

        class Forgot(Flow[Item]):
            inner = Source.bind()

            edges = (edge(START).to(inner), edge(inner).to(EXIT))


def test_unknown_binding_field_is_rejected():
    with pytest.raises(TypeError, match="has no require\\(\\) named 'nope'"):

        class Typo(Flow[Item]):
            @step
            async def make(self) -> Item:
                return Item(value=1)

            inner = Source.bind(seed=make, nope=make)

            edges = (edge(START).to(make), edge(make).to(inner), edge(inner).to(EXIT))


def test_binding_to_a_producer_from_elsewhere_is_rejected():
    with pytest.raises(TypeError, match="not declared on"):

        class Foreign(Flow[Item]):
            inner = Source.bind(seed=Closed.only)

            edges = (edge(START).to(inner), edge(inner).to(EXIT))


def test_binding_the_wrong_model_is_rejected():
    with pytest.raises(TypeError, match="wants Item but"):

        class Mismatch(Flow[Item]):
            @step
            async def make(self) -> Other:
                return Other(text="x")

            inner = Source.bind(seed=make)

            edges = (edge(START).to(make), edge(make).to(inner), edge(inner).to(EXIT))


def test_binding_to_something_that_persists_nothing_is_rejected():
    """A step with no return annotation persists nothing, so there is no value to read —
    the requirement would silently be None."""
    with pytest.raises(TypeError, match="wants Item but"):

        class Empty(Flow[Item]):
            @step
            async def nothing(self) -> None:
                return None

            inner = Source.bind(seed=nothing)

            edges = (edge(START).to(nothing), edge(nothing).to(inner), edge(inner).to(EXIT))


def test_a_requirement_may_be_forwarded_to_a_child():
    class Forward(Flow[Item]):
        seed = require(Item)
        inner = Source.bind(seed=seed)

        edges = (edge(START).to(inner), edge(inner).to(EXIT))

    assert set(Forward.plan().requirements) == {"seed"}


# --- one producer, one name -----------------------------------------------------------


def test_two_names_for_one_flow_ref_is_rejected():
    with pytest.raises(TypeError, match="are the same"):

        class Aliased(Flow[Item]):
            @step
            async def make(self) -> Item:
                return Item(value=1)

            inner = Source.bind(seed=make)
            same = inner

            edges = (edge(START).to(make), edge(make).to(inner), edge(inner).to(EXIT))


def test_two_names_for_one_step_is_rejected():
    with pytest.raises(TypeError, match="already belongs to"):

        class AliasedStep(Flow[Item]):
            @step
            async def make(self) -> Item:
                return Item(value=1)

            same = make

            edges = (edge(START).to(make), edge(make).to(EXIT))


def test_a_step_named_like_the_cursor_is_rejected():
    with pytest.raises(TypeError, match="reserved for the loop cursor"):

        class Clash(Flow[Item]):
            @step
            async def _loop_cursor(self) -> Item:
                return Item(value=1)

            edges = (edge(START).to(_loop_cursor), edge(_loop_cursor).to(EXIT))


# --- shape ----------------------------------------------------------------------------


def test_a_flow_needs_nodes():
    with pytest.raises(TypeError, match="a flow needs nodes"):

        class Nothing(Flow[Item]):
            pass


def test_a_flow_must_declare_edges():
    with pytest.raises(TypeError, match="a flow declares its order"):

        class NoEdges(Flow[Item]):
            @step
            async def only(self) -> Item:
                return Item(value=1)


def test_a_flow_may_not_define_init():
    with pytest.raises(TypeError, match="defines __init__"):

        class Constructed(Flow[Item]):
            def __init__(self, *, tag: str) -> None: ...

            @step
            async def only(self) -> Item:
                return Item(value=1)

            edges = (edge(START).to(only), edge(only).to(EXIT))


def test_max_steps_must_be_positive():
    with pytest.raises(TypeError, match="max_steps must be at least 1"):

        class Fused(Flow[Item]):
            max_steps = 0

            @step
            async def only(self) -> Item:
                return Item(value=1)

            edges = (edge(START).to(only), edge(only).to(EXIT))


# --- the output contract --------------------------------------------------------------


def test_declared_output_must_match_the_terminal():
    with pytest.raises(TypeError, match="declared Flow\\[Item\\] but its terminals produce Other"):

        class Wrong(Flow[Item]):
            @step
            async def only(self) -> Other:
                return Other(text="x")

            edges = (edge(START).to(only), edge(only).to(EXIT))


def test_terminals_must_agree_with_each_other():
    with pytest.raises(TypeError, match="don't agree on what this flow produces"):

        class Split(Flow):
            @step
            async def a(self) -> Item:
                return Item(value=1)

            @step
            async def b(self) -> Other:
                return Other(text="x")

            edges = (
                edge(START).to(a),
                edge(a).when(lambda i: i.value > 0).to(EXIT).otherwise(b),
                edge(b).to(EXIT),
            )


def test_several_terminals_of_one_model_are_fine():
    class Branching(Flow[Item]):
        @step
        async def a(self) -> Item:
            return Item(value=1)

        @step
        async def b(self) -> Item:
            return Item(value=2)

        edges = (
            edge(START).to(a),
            edge(a).when(lambda i: i.value > 5).to(EXIT).otherwise(b),
            edge(b).to(EXIT),
        )

    assert sorted(Branching.plan().graph.terminals()) == ["a", "b"]
    assert Branching.output_model() is Item


def test_an_undeclared_output_type_is_taken_from_the_terminals():
    class Bare(Flow):
        @step
        async def only(self) -> Item:
            return Item(value=1)

        edges = (edge(START).to(only), edge(only).to(EXIT))

    assert Bare.output_model() is Item


def test_a_flow_whose_terminal_persists_nothing_produces_nothing():
    class Silent(Flow):
        @step
        async def only(self) -> None:
            return None

        edges = (edge(START).to(only), edge(only).to(EXIT))

    assert Silent.output_model() is None
