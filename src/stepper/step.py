"""Step: the unit of work, and the `@step` / `depends` wiring it's built from.

`@step` turns an async `Stage` method into a `Step[R]` handle. List a handle in a
stage's `steps = (...)` to run it, or pass it to `depends(...)` to feed its
persisted output into another step (even a step on a different stage). `R` is the
step's return type, so `depends()` types the consuming parameter correctly.
"""

from __future__ import annotations

from inspect import signature
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Coroutine, Generic, TypeVar, cast

if TYPE_CHECKING:
    from stepper.stage import Stage

R = TypeVar("R")


class Step(Generic[R]):
    """A step handle — what `@step` gives you. Holds the underlying async fn, its
    name, and the model to persist/fetch its value as (inferred from the return
    annotation). `owner` is the stage that lists it."""

    def __init__(self, fn: Callable[..., Awaitable[Any]]) -> None:
        self.fn = fn
        self.name = fn.__name__
        self.model: Any = fn.__annotations__.get("return")
        self.owner: "type[Stage] | None" = None

    def claim(self, stage: "type[Stage]") -> None:
        if self.owner is not None:
            raise TypeError(f"step {self.name!r} already belongs to {self.owner.__name__}, can't also be in {stage.__name__}.")
        self.owner = stage

    def dependencies(self) -> dict[str, "Step[Any]"]:
        """Map each parameter to the step it depends on — its `depends(...)` default.
        Returns `{param_name: dependency_step}`."""
        wiring: dict[str, "Step[Any]"] = {}
        for name, param in signature(self.fn).parameters.items():
            if name == "self":
                continue
            if not isinstance(param.default, Step):
                raise TypeError(f"step {self.name!r}: param {name!r} must be wired with depends(...).")
            wiring[name] = param.default
        return wiring

    def get_owner(self) -> "type[Stage]":
        if self.owner is None:
            raise TypeError(f"step {self.name!r} is not claimed by any stage.")
        return self.owner


def step(fn: Callable[..., Coroutine[Any, Any, R]]) -> Step[R]:
    """Turn an async method into a step you can list in a `Stage` and wire with `depends()`."""
    return Step(fn)


def depends(producer: Step[R]) -> R:
    """Inject a producing step's output as a parameter default, typed as its return
    type. The `Step` is the wiring marker; the framework passes in the persisted value."""
    return cast("R", producer)
