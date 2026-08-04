"""The whole point of this module is its first line.

Under PEP 563 every annotation is a *string* at run time, so a step reading raw
`fn.__annotations__` would infer `model == "Item"` — a str — and silently break
`TypeAdapter(model)` and `model.__name__`. `Step` resolves through `get_type_hints`
instead. `src/play.py` uses this import, so it's the shape most consumers will write.
"""

from __future__ import annotations

from pydantic import BaseModel

from stepper import EXIT, START, Flow, depends, edge, require, step


class Item(BaseModel):
    value: int


class Doubler(Flow[Item]):
    seed = require(Item)

    @step
    async def double(self, item=depends(seed)) -> Item:
        return Item(value=item.value * 2)

    edges = (edge(START).to(double), edge(double).to(EXIT))


class Root(Flow[str]):
    @step
    async def start(self) -> Item:
        return Item(value=3)

    inner = Doubler.bind(seed=start)

    @step
    async def note(self, item=depends(inner)) -> str:
        return f"got {item.value}"

    edges = (
        edge(START).to(start),
        edge(start).to(inner),
        edge(inner).to(note),
        edge(note).to(EXIT),
    )


def test_a_stringized_annotation_resolves_to_the_real_model():
    assert Root.producers()["start"].model is Item
    assert Doubler.producers()["double"].model is Item
    assert Root.producers()["note"].model is str


def test_the_declared_output_type_still_matches_its_terminals():
    assert Doubler.output_model() is Item
    assert Root.output_model() is str


def test_a_run_round_trips_under_pep_563(run, mem):
    assert run(Root().run(persist_service=mem)) == "got 6"
    assert mem.fetch("root/inner/double", Item) == Item(value=6)


def test_the_return_check_still_knows_the_real_type(run, mem):
    """`_check` builds a TypeAdapter from the model — a str would make that meaningless."""
    import pytest

    class Liar(Flow[Item]):
        @step
        async def make(self) -> Item:
            return "not an item"  # type: ignore[return-value]

        edges = (edge(START).to(make), edge(make).to(EXIT))

    with pytest.raises(TypeError, match="declared -> Item but returned str"):
        run(Liar().run(persist_service=mem))
