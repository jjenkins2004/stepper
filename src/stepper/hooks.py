"""Lifecycle hooks: wrap every node a run executes so callers can add telemetry or side
effects (tracing spans, metrics, actions) without the framework depending on any tracing
library. Pass a `Hooks` implementation to the flow you run and it, and everything mounted
under it, is wrapped; pass none and nothing is.

Each hook returns a context manager entered around the work: code before `yield` runs
before the node, code after runs when it finishes or raises — the natural place to open
and close a span. A node is identified by its `path`, which is also the key it persists
under, so a span and its artifact carry the same name.

A step's output only exists *after* the step runs (after your `yield`), so to capture
it you yield a `StepReport`: the framework fills it via `set_output` once the step has
run and persisted, and your after-`yield` code reads `report.output`. The framework
only ever touches its own `StepReport` type — nothing tracing-specific.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Protocol


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
    """Wrap step and flow execution. Implement with `@contextmanager` methods (or any
    object whose `step`/`flow` return a context manager) to emit spans/metrics/actions.
    Yield a `StepReport` from `step` to receive the step's output."""

    def step(
        self, *, path: str, input_type: str, output_type: str
    ) -> AbstractContextManager[StepReport | None]:
        """Wrap one step.

        Args:
            path: The step's path — `"push/tiktok/upload/attempt"`. Also its persist key.
            input_type: Comma-joined names of the models the step reads via `depends()`,
                or "None" if it takes no inputs.
            output_type: Name of the model the step returns, or "None" if it persists nothing.

        Yield a `StepReport` to receive the step's output after it runs, or yield nothing.
        """
        ...

    def flow(self, *, path: str, node_count: int) -> AbstractContextManager[Any]:
        """Wrap one flow run. `node_count` is how many nodes the flow contains."""
        ...
