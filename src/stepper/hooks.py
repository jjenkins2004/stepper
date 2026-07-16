"""Lifecycle hooks: wrap each step and each stage run so callers can add telemetry or
side effects (tracing spans, metrics, actions) without the framework depending on any
tracing library. Pass a `Hooks` implementation to `Pipeline` (or a `Stage`); the
default is a no-op.

Each hook returns a context manager entered around the work: code before `yield` runs
before the step/stage, code after runs when it finishes or raises — the natural place
to open and close a span.

A step's output only exists *after* the step runs (after your `yield`), so to capture
it you yield a `StepReport`: the framework fills it via `set_output` once the step has
run and persisted, and your after-`yield` code reads `report.output`. The framework
only ever touches its own `StepReport` type — nothing tracing-specific.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any, Iterator, Protocol


class StepReport:
    """Carries a step's output back to your hook. Yield one from your `step` hook; the
    framework fills it after the step runs and persists (you never call `set_output`
    yourself). Read `.output` — guarded by `.has_output` — in the code after `yield`."""

    def __init__(self) -> None:
        self._value: Any = None
        self._has = False

    def set_output(self, value: Any) -> None:
        """Called by the framework with the step's return value."""
        self._value = value
        self._has = True

    @property
    def has_output(self) -> bool:
        return self._has

    @property
    def output(self) -> Any:
        return self._value


class Hooks(Protocol):
    """Wrap step and stage execution. Implement with `@contextmanager` methods (or any
    object whose `step`/`stage` return a context manager) to emit spans/metrics/actions.
    Yield a `StepReport` from `step` to receive the step's output."""

    def step(
        self, *, stage_name: str, step_name: str, input_type: str, output_type: str
    ) -> AbstractContextManager[StepReport | None]: ...

    def stage(self, *, stage_name: str, step_count: int) -> AbstractContextManager[Any]: ...


class NoOpHooks:
    """Default `Hooks`: every hook is an empty context manager."""

    @contextmanager
    def step(
        self, *, stage_name: str, step_name: str, input_type: str, output_type: str
    ) -> Iterator[StepReport]:
        yield StepReport()

    def stage(self, *, stage_name: str, step_count: int) -> AbstractContextManager[Any]:
        return nullcontext()
