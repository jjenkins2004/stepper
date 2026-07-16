"""Reusable models + stages for stepper tests.

No ``from __future__ import annotations`` — ``Step`` reads raw ``fn.__annotations__``,
so stringized annotations would break ``.model`` inference and the runners.

Persist keys produced: ``A/prod`` (json), ``B/consume`` (json), ``B/note`` (txt).
``BStage`` exercises a cross-stage dep (``consume`` <- ``AStage.prod``) and a
same-stage dep (``note`` <- ``consume``).
"""

from pydantic import BaseModel

from stepper.stage import Stage
from stepper.step import depends, step


class Item(BaseModel):
    value: int


class AStage(Stage):
    @step
    async def prod(self) -> Item:
        return Item(value=1)

    steps = (prod,)


class BStage(Stage):
    @step
    async def consume(self, item=depends(AStage.prod)) -> Item:
        return Item(value=item.value + 1)

    @step
    async def note(self, item=depends(consume)) -> str:
        return f"got {item.value}"

    steps = (consume, note)
