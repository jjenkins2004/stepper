"""Step: the unit of work, and the `@step` / `depends` wiring it's built from.

`@step` turns an async `Stage` method into a `Step[R]` handle. List a handle in a
stage's `steps = (...)` to run it, or pass it to `depends(...)` to feed its
persisted output into another step (even a step on a different stage). `R` is the
step's return type, so `depends()` types the consuming parameter correctly.
"""

from __future__ import annotations

from inspect import signature
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Coroutine, Generic, TypeVar, cast, get_type_hints, overload

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
        `optional_dependencies` for which params are optional). A dep written as a name
        is resolved here, against the stage that claimed this step."""
        wiring: dict[str, "Step[Any]"] = {}
        for name, param in signature(self.fn).parameters.items():
            if name == "self":
                continue
            dep = _as_step(param.default)
            if dep is None:
                raise TypeError(f"step {self.name!r}: param {name!r} must be wired with depends(...) or optional_depends(...).")
            wiring[name] = self._resolve(dep, name)
        return wiring

    def _resolve(self, dep: "Step[Any] | _LazyRef", param: str) -> "Step[Any]":
        """A `_LazyRef` looked up by name among the owning stage's steps. Only a name can
        point *backwards* in a loop, so this is where a back-edge dep becomes a real Step."""
        if isinstance(dep, Step):
            return dep
        for candidate in self.get_owner().steps:
            if candidate.name == dep.name:
                return candidate
        raise TypeError(
            f"step {self.name!r}: param {param!r} depends on {dep.name!r}, "
            f"which is not a step on {self.get_owner().__name__}."
        )

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


class _LazyRef:
    """A dep written as a step *name* instead of the `Step` itself, resolved against the
    owning stage once the class exists. The escape hatch for a reference that has to point
    backwards: in an edge-driven stage the back edge's producer is defined *below* its consumer
    in the class body, so there is no object to pass yet — only a name."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name


@overload
def depends(producer: Step[R]) -> R: ...
@overload
def depends(producer: str) -> Any: ...
def depends(producer: Step[R] | str) -> Any:
    """Inject a producing step's output as a parameter default, typed as its return
    type. The `Step` is the wiring marker; the framework passes in the persisted value.

    Pass a step *name* instead when the producer is defined later in the class body —
    the back edge of an edge-driven stage. A name carries no type, so annotate the parameter
    yourself: `prev: Analysis = depends("analysis")`."""
    return cast("R", _LazyRef(producer) if isinstance(producer, str) else producer)


class _OptionalDep:
    """Wiring marker from `optional_depends()`: the producer (a `Step`, or a `_LazyRef` to
    one) plus the "missing -> None" intent. `Step.dependencies()` unwraps it to the `Step`
    (so it schedules like any dep); `Step.optional_dependencies()` reports its param name."""

    __slots__ = ("step",)

    def __init__(self, step: "Step[Any] | _LazyRef") -> None:
        self.step = step


def _as_step(default: Any) -> "Step[Any] | _LazyRef | None":
    """The producer behind a `depends()` / `optional_depends()` default — a `Step`, or a
    `_LazyRef` still to be resolved — or None if the parameter wasn't wired with either."""
    if isinstance(default, (Step, _LazyRef)):
        return default
    if isinstance(default, _OptionalDep):
        return default.step
    return None


@overload
def optional_depends(producer: Step[R]) -> R | None: ...
@overload
def optional_depends(producer: str) -> Any: ...
def optional_depends(producer: Step[R] | str) -> Any:
    """Like `depends`, but typed `R | None`: if the producer's value isn't persisted when
    this step runs, the parameter gets None instead of the run raising. Scheduling is
    unchanged — an in-stage producer is still waited for, and if it *fails* this step is
    still skipped; optionality only turns a missing persisted value into None (e.g. a
    cross-stage producer that didn't run, or a step that persisted nothing).

    This is the usual way to read across a loop's back edge: the producer is defined below
    the consumer, so pass its *name*, and the first pass — when it hasn't run yet — reads
    back as None."""
    return cast("R | None", _OptionalDep(_LazyRef(producer) if isinstance(producer, str) else producer))
