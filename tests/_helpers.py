"""Reusable models and flows.

No ``from __future__ import annotations`` — ``Step`` resolves return annotations through
``get_type_hints``, which works either way, but keeping them real keeps these readable.

``RootFlow`` mounts ``DoubleFlow`` twice, which is the shape most tests care about: one
class, two mounts, two key namespaces, bound to different producers.
"""

from pydantic import BaseModel

from stepper import EXIT, START, Flow, depends, edge, require, step


class Item(BaseModel):
    value: int


class DoubleFlow(Flow[Item]):
    """Open: needs an Item, returns twice it."""

    seed = require(Item)

    @step
    async def double(self, item=depends(seed)) -> Item:
        return Item(value=item.value * 2)

    edges = (edge(START).to(double), edge(double).to(EXIT))


class RootFlow(Flow[Item]):
    """Closed: two mounts of DoubleFlow over the same seed, summed."""

    @step
    async def start(self) -> Item:
        return Item(value=1)

    left = DoubleFlow.bind(seed=start)
    right = DoubleFlow.bind(seed=left)

    @step
    async def total(self, a=depends(left), b=depends(right)) -> Item:
        return Item(value=a.value + b.value)

    edges = (
        edge(START).to(start),
        edge(start).to(left),
        edge(left).to(right),
        edge(right).to(total),
        edge(total).to(EXIT),
    )
