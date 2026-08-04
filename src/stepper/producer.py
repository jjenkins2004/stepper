"""Producer: anything with one typed output that `depends()` can read.

There are exactly three, and a step consuming one can't tell them apart:

- **`Step`** — an output it computes itself.
- **`FlowRef`** (`SomeFlow.bind(...)`) — an output its subtree computes.
- **`Requirement`** (`require(Listing)`) — an output with no producer, supplied by whoever
  mounts the flow that declares it.

A producer is a *class attribute* of the flow that declares it, which is not an accident: a
`depends()` is a parameter default, evaluated while the class body runs, so it can only
point at something already in that body. There is no `self` yet, and never will be. That is
the whole reason a flow's inputs are declared with `require()` rather than passed to a
constructor — a constructor argument arrives too late for any step to name it.

It's also what confines wiring to one flow. Every `depends()` names an attribute of the
class it's written in, so nothing can reach into, or out of, a flow it doesn't own.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

R = TypeVar("R")


class Producer(Generic[R]):
    """One typed output, under the name the flow's class body binds it to."""

    name: str = ""
    _model: Any = None

    def __set_name__(self, owner: type, name: str) -> None:
        # Python hands us the attribute name once the class body finishes, so a producer is
        # never told what it's called. This is the *only* source of a node's name: a step
        # bound as `check = step(fn)` is `check`, whatever `fn` happened to be called, so
        # the path it persists under and the name a `depends()` resolves can't disagree.
        self.name = name

    @property
    def model(self) -> Any:
        """The model this producer's value is persisted and fetched as, or None when it
        persists nothing."""
        return self._model


class Requirement(Producer[R]):
    """A flow's input: an output it declares but does not produce.

    Steps inside read it exactly as they read anything else (`depends(listing)`); the flow
    that mounts this one says where the value comes from (`Cls.bind(listing=...)`). A flow
    with no requirements is *closed* and runs on its own; one with requirements is *open*
    and runs wherever they're supplied.
    """

    def __init__(self, model: type[R]) -> None:
        self._model = model

    def __repr__(self) -> str:
        return f"require({getattr(self._model, '__name__', self._model)})"


def require(model: type[R]) -> Requirement[R]:
    """Declare an input this flow needs from outside. Typed, so `depends()` on it gives
    the consuming parameter the model's type."""
    return Requirement(model)


__all__ = ["Producer", "Requirement", "require"]
