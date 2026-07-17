"""Step: the unit of work, and the `@step` / `depends` wiring it's built from.

`@step` turns an async `Stage` method into a `Step[R]` handle. List a handle in a
stage's `steps = (...)` to run it, or pass it to `depends(...)` to feed its
persisted output into another step (even a step on a different stage). `R` is the
step's return type, so `depends()` types the consuming parameter correctly.
"""

from __future__ import annotations

from inspect import signature
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Coroutine, Generic, TypeVar, cast, get_type_hints

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
        # Resolve the return annotation to a real type. get_type_hints evaluates
        # PEP 563 / `from __future__ import annotations` string annotations; reading
        # raw __annotations__ leaves them as str, so model="Draft" (a str) silently
        # breaks TypeAdapter(model) / model.__name__ at run time. get_type_hints
        # reports a `-> None` return as NoneType, so normalize it back to None to
        # preserve the "no model => don't persist" contract stage.py relies on.
        ret = get_type_hints(fn).get("return")
        self.model: Any = None if ret is type(None) else ret
        self.owner: "type[Stage] | None" = None

    def claim(self, stage: "type[Stage]") -> None:
        if self.owner is not None:
            raise TypeError(f"step {self.name!r} already belongs to {self.owner.__name__}, can't also be in {stage.__name__}.")
        self.owner = stage

    def dependencies(self) -> dict[str, "Step[Any]"]:
        """Map each parameter to the step it depends on — its `depends(...)` /
        `optional_depends(...)` default. Returns `{param_name: dependency_step}`; an
        optional dep is unwrapped to the same `Step`, so it schedules identically (see
        `optional_dependencies` for which params are optional)."""
        wiring: dict[str, "Step[Any]"] = {}
        for name, param in signature(self.fn).parameters.items():
            if name == "self":
                continue
            dep = _as_step(param.default)
            if dep is None:
                raise TypeError(f"step {self.name!r}: param {name!r} must be wired with depends(...) or optional_depends(...).")
            wiring[name] = dep
        return wiring

    def optional_dependencies(self) -> set[str]:
        """Names of the params wired with `optional_depends(...)` — a subset of
        `dependencies()`. For these, a missing persisted value is read back as None
        instead of raising."""
        return {
            name for name, param in signature(self.fn).parameters.items()
            if isinstance(param.default, _OptionalDep)
        }

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


class _OptionalDep:
    """Wiring marker from `optional_depends()`: the producer `Step` plus the "missing ->
    None" intent. `Step.dependencies()` unwraps it to the `Step` (so it schedules like any
    dep); `Step.optional_dependencies()` reports its param name."""

    __slots__ = ("step",)

    def __init__(self, step: Step[Any]) -> None:
        self.step = step


def _as_step(default: Any) -> "Step[Any] | None":
    """The `Step` behind a `depends()` / `optional_depends()` default, or None if the
    parameter wasn't wired with either."""
    if isinstance(default, Step):
        return default
    if isinstance(default, _OptionalDep):
        return default.step
    return None


def optional_depends(producer: Step[R]) -> R | None:
    """Like `depends`, but typed `R | None`: if the producer's value isn't persisted when
    this step runs, the parameter gets None instead of the run raising. Scheduling is
    unchanged — an in-stage producer is still waited for, and if it *fails* this step is
    still skipped; optionality only turns a missing persisted value into None (e.g. a
    cross-stage producer that didn't run, or a step that persisted nothing)."""
    return cast("R | None", _OptionalDep(producer))
