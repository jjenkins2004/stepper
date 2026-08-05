"""Step: the leaf node, and the `@step` / `depends` wiring it's built from.

`@step` turns an async `Flow` method into a `Step[R]` handle — a `Producer`, like a nested
flow or a `require()` is. Wire an input by defaulting a parameter to `depends(x)`, where
`x` is any producer declared on the *same class*: another step, a child flow, or one of the
flow's own requirements. Nothing else is in scope, which is what keeps a flow's wiring
inside the flow.

A step is a `Node` exactly as a flow is, with the same `run()` and the same path: it
persists under its own path and, being a leaf, refuses an address beneath it. Running one
is fetch inputs, call the function, persist the result — the whole of it.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack
from inspect import signature
from time import perf_counter
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Coroutine, TypeVar, cast, get_type_hints, overload

from pydantic import TypeAdapter, ValidationError

from stepper.node import Context, Node
from stepper.persist import nested_persistable_path
from stepper.producer import Producer
from stepper.step_logging import format_step_end, format_step_fail, format_step_start

if TYPE_CHECKING:
    from stepper.flow import Flow

R = TypeVar("R")

_LOGGER = logging.getLogger(__name__)


class Step(Node, Producer[R]):
    """A step handle — what `@step` gives you. Holds the underlying async fn, its name, and
    the model to persist/fetch its value as (inferred from the return annotation). `owner`
    is the flow class whose body declares it."""

    def __init__(self, fn: Callable[..., Awaitable[Any]]) -> None:
        super().__init__()
        self.fn = fn
        self.name = fn.__name__
        # Resolve the return annotation to a real type. get_type_hints evaluates
        # PEP 563 / `from __future__ import annotations` string annotations; reading
        # raw __annotations__ leaves them as str, so model="Draft" (a str) silently
        # breaks TypeAdapter(model) / model.__name__ at run time. get_type_hints
        # reports a `-> None` return as NoneType, so normalize it back to None to
        # preserve the "no model => don't persist" contract the runner relies on.
        ret = get_type_hints(fn).get("return")
        self._model: Any = None if ret is type(None) else ret
        # A Persistable only gets its hooks run when it IS the declared model, so one
        # buried in a plain model would lose its side-artifacts with nothing raised.
        # Every model that crosses persist starts as some step's return annotation —
        # a flow's output must equal a terminal's, and a require() must match the
        # producer bound to it — so checking here covers all three.
        buried = nested_persistable_path(self._model)
        if buried is not None:
            raise TypeError(
                f"step {self.name!r} returns {getattr(self._model, '__name__', self._model)}, "
                f"which holds a Persistable at {buried}. Only the declared model's hooks "
                f"run, so that one's side-artifacts would be dropped without an error. "
                f"Return the Persistable itself, or make the outer model a Persistable "
                f"that forwards on_persist/on_fetch to it."
            )
        self._adapter: TypeAdapter[Any] | None = None
        self.owner: "type[Flow] | None" = None

    # --- declaration ------------------------------------------------------------------

    def claim(self, owner: "type[Flow]") -> None:
        if self.owner is not None:
            raise TypeError(
                f"step {self.name!r} already belongs to {self.owner.__name__}, "
                f"can't also be in {owner.__name__}."
            )
        self.owner = owner

    def __repr__(self) -> str:
        owner = self.owner.__name__ + "." if self.owner is not None else ""
        return f"step({owner}{self.name})"

    def get_owner(self) -> "type[Flow]":
        if self.owner is None:
            raise TypeError(f"step {self.name!r} is not declared in any flow.")
        return self.owner

    def dependencies(self) -> dict[str, Producer[Any]]:
        """Map each parameter to the producer it reads — its `depends(...)` /
        `optional_depends(...)` default. An optional dep is unwrapped to the same producer,
        so it schedules identically (see `optional_dependencies` for which params are
        optional). A dep written as a name is resolved here, against the flow that claimed
        this step."""
        wiring: dict[str, Producer[Any]] = {}
        for name, param in signature(self.fn).parameters.items():
            if name == "self":
                continue
            dep = _as_producer(param.default)
            if dep is None:
                raise TypeError(
                    f"step {self.name!r}: param {name!r} must be wired with depends(...) "
                    f"or optional_depends(...)."
                )
            wiring[name] = self._resolve(dep, name)
        return wiring

    def _resolve(self, dep: "Producer[Any] | _LazyRef", param: str) -> Producer[Any]:
        """A `_LazyRef` looked up by name among the owning flow's producers. Only a name
        can point *backwards* in a loop, so this is where a back-edge dep becomes real."""
        if isinstance(dep, Producer):
            return dep
        found = self.get_owner().producers().get(dep.name)
        if found is None:
            raise TypeError(
                f"step {self.name!r}: param {param!r} depends on {dep.name!r}, which is not "
                f"declared on {self.get_owner().__name__}."
            )
        return found

    def optional_dependencies(self) -> set[str]:
        """Names of the params wired with `optional_depends(...)` — a subset of
        `dependencies()`. For these, a missing persisted value is read back as None
        instead of raising."""
        return {
            name for name, param in signature(self.fn).parameters.items()
            if isinstance(param.default, _OptionalDep)
        }

    # --- mounting ---------------------------------------------------------------------

    def _bind(self, *, path: str, ctx: Context, parent: "Flow") -> "Step[R]":
        twin: Step[R] = Step.__new__(Step)
        Node.__init__(twin)
        twin.fn = self.fn
        twin.name = self.name
        twin._model = self._model
        twin._adapter = self._adapter
        twin.owner = self.owner
        twin._path = path
        twin._ctx = ctx
        twin._parent = parent
        return twin

    # --- running ----------------------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        return {
            "path": self._path or self.name,
            "kind": "step",
            "output": self.model.__name__ if self.model is not None else None,
        }

    def _check(self, result: Any) -> None:
        """A step's return annotation is a promise about what lands on disk, so hold it to
        that before writing. Without this the backend serializes a mismatch to `{}` and the
        failure surfaces later, in whichever step tried to read it — or not at all."""
        if self._adapter is None:
            self._adapter = TypeAdapter(self.model)
        try:
            self._adapter.validate_python(result)
        except ValidationError as exc:
            raise TypeError(
                f"{self.path}: declared -> {self.model.__name__} but returned "
                f"{type(result).__name__}."
            ) from exc

    async def run(self, target: str | None = None, *, follow_edges: bool = False) -> Any:
        """Fetch this step's inputs, run it, persist the result. A `target` is an error:
        a step is a leaf, so there is nothing beneath it to address."""
        if target:
            raise ValueError(
                f"{self.path}: {target!r} addresses a node inside a step, and a step is a leaf."
            )
        if follow_edges:
            raise ValueError(f"{self.path}: follow_edges needs a flow that declares edges.")

        ctx = self.ctx
        flow = self._parent
        assert flow is not None                 # a bound step always has its flow
        deps = self.dependencies()
        optional = self.optional_dependencies()
        input_type = ", ".join(
            dep.model.__name__ if dep.model is not None else "None" for dep in deps.values()
        ) or "None"
        output_type = self.model.__name__ if self.model is not None else "None"
        _LOGGER.info(format_step_start(step_name=self.name, input_type=input_type, output_type=output_type))
        started = perf_counter()
        try:
            with ExitStack() as stack:
                # Enter every hook's step() in order (ExitStack exits them in reverse).
                # One hook behaves exactly as one always has; () is a clean no-op. A raise
                # in __enter__/__exit__ unwinds the already-entered hooks with it.
                reports = [
                    stack.enter_context(
                        hook.step(
                            path=self._path,
                            input_type=input_type,
                            output_type=output_type,
                        )
                    )
                    for hook in ctx.hooks
                ]
                # Grab each declared input. Where a value lives is the flow's business — it
                # knows what its own producers resolved to. A required dep with nothing
                # persisted raises; an optional one reads back as None.
                inputs: dict[str, Any] = {}
                for name, dep in deps.items():
                    if dep.model is None:
                        # Producer persists nothing, so there is nothing to fetch — the dep
                        # is pure ordering and the parameter is None. Typed that way too,
                        # since `depends()` on a `Step[None]` is None.
                        inputs[name] = None
                        continue
                    try:
                        inputs[name] = ctx.persist.fetch(ctx.key(flow.key_of(dep)), dep.model)
                    except FileNotFoundError:
                        if name not in optional:
                            raise
                        inputs[name] = None

                # Run it once. `self` is the bound flow, so a step can read whatever its
                # flow knows — its path, for instance.
                result = await self.fn(flow, **inputs)

                # Persist under this node's own path, then hand the value to each hook's
                # StepReport (skipping any hook that yielded None).
                if self.model is not None:
                    self._check(result)
                    ctx.persist.persist(ctx.key(self._path), result, self.model)
                    for report in reports:
                        if report is not None:
                            report.set_output(result)
        except Exception as exc:
            elapsed_ms = int((perf_counter() - started) * 1000)
            _LOGGER.exception(
                format_step_fail(step_name=self.name, elapsed_ms=elapsed_ms, error_type=type(exc).__name__)
            )
            raise

        elapsed_ms = int((perf_counter() - started) * 1000)
        _LOGGER.info(
            format_step_end(step_name=self.name, elapsed_ms=elapsed_ms, output_type=type(result).__name__)
        )
        return result


def step(fn: Callable[..., Coroutine[Any, Any, R]]) -> Step[R]:
    """Turn an async method into a step you can declare in a `Flow` and wire with
    `depends()`."""
    return Step(fn)


class _LazyRef:
    """A dep written as a *name* instead of the producer itself, resolved against the
    owning flow once the class exists. The escape hatch for a reference that has to point
    backwards: in an edge-driven flow the back edge's producer is declared *below* its
    consumer in the class body, so there is no object to pass yet — only a name."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name


@overload
def depends(producer: Producer[R]) -> R: ...
@overload
def depends(producer: str) -> Any: ...
def depends(producer: Producer[R] | str) -> Any:
    """Read a producer's output as a parameter default, typed as that producer's model.
    The producer is the wiring marker; the framework passes in the persisted value.

    It must be declared on the same class — a step, a nested flow, or one of this flow's
    `require()` inputs. Pass a *name* instead when it's declared later in the class body,
    which is the back edge of an edge-driven flow; a name carries no type, so annotate the
    parameter yourself: `prev: Analysis = depends("analysis")`."""
    return cast("R", _LazyRef(producer) if isinstance(producer, str) else producer)


class _OptionalDep:
    """Wiring marker from `optional_depends()`: the producer (or a `_LazyRef` to one) plus
    the "missing -> None" intent. `Step.dependencies()` unwraps it (so it schedules like
    any dep); `Step.optional_dependencies()` reports its param name."""

    __slots__ = ("producer",)

    def __init__(self, producer: "Producer[Any] | _LazyRef") -> None:
        self.producer = producer


def _as_producer(default: Any) -> "Producer[Any] | _LazyRef | None":
    """The producer behind a `depends()` / `optional_depends()` default, or None if the
    parameter wasn't wired with either."""
    if isinstance(default, (Producer, _LazyRef)):
        return default
    if isinstance(default, _OptionalDep):
        return default.producer
    return None


@overload
def optional_depends(producer: Producer[R]) -> R | None: ...
@overload
def optional_depends(producer: str) -> Any: ...
def optional_depends(producer: Producer[R] | str) -> Any:
    """Like `depends`, but typed `R | None`: if the producer's value isn't persisted when
    this step runs, the parameter gets None instead of the run raising. Scheduling is
    unchanged — the producer is still waited for, and if it *fails* this step is still
    skipped; optionality only turns a missing persisted value into None.

    This is the usual way to read across a loop's back edge: the producer is declared below
    the consumer, so pass its *name*, and the first pass — when it hasn't run yet — reads
    back as None."""
    return cast("R | None", _OptionalDep(_LazyRef(producer) if isinstance(producer, str) else producer))
