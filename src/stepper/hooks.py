"""Lifecycle hooks: wrap each step and each stage run so callers can add telemetry or
side effects (tracing spans, metrics, actions) without the framework depending on any
tracing library. Pass a `Hooks` implementation to `Pipeline` (or a `Stage`); the
default is a no-op.

Each hook returns a context manager entered around the work: code before `yield` runs
before the step/stage, code after runs when it finishes or raises — the natural place
to open and close a span.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any, Protocol


class Hooks(Protocol):
    """Wrap step and stage execution. Implement with `@contextmanager` methods (or any
    object whose `step`/`stage` return a context manager) to emit spans/metrics/actions."""

    def step(
        self, *, stage_name: str, step_name: str, input_type: str, output_type: str
    ) -> AbstractContextManager[Any]: ...

    def stage(self, *, stage_name: str, step_count: int) -> AbstractContextManager[Any]: ...


class NoOpHooks:
    """Default `Hooks`: every hook is an empty context manager."""

    def step(
        self, *, stage_name: str, step_name: str, input_type: str, output_type: str
    ) -> AbstractContextManager[Any]:
        return nullcontext()

    def stage(self, *, stage_name: str, step_count: int) -> AbstractContextManager[Any]:
        return nullcontext()
