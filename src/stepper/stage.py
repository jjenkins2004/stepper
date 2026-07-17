"""Stage: a set of steps sharing one PersistService, run by a dependency scheduler.

Subclass `Stage`, mark async methods with `@step`, and list them in `steps = (...)`.
The tuple is just the *membership* set — which steps belong to the stage. Run order
comes from each step's `depends()` markers: `Scheduler` turns the tuple into a
validated DAG, and `run_steps()` follows it — launching every step whose upstreams
have finished, concurrently, so independent steps run at the same time.

Each step persists its return value under `<stage_name>/<step>`; wire an input by
defaulting a parameter to `depends(producer_step)`, which fetches that step's
persisted value (typed as its return type). A `depends()` may point at a step on
another stage — that value is already on disk from the earlier stage, so it's an
input, not a scheduling edge within this stage.

Add tracing/telemetry by passing a `Hooks` implementation (default: no-op); each step
and each stage run is wrapped in the matching hook context manager.
"""

import logging
from time import perf_counter
from typing import Any, Callable, ClassVar, Coroutine

from stepper.hooks import Hooks, NoOpHooks
from stepper.persist import PersistService
from stepper.scheduler import Scheduler
from stepper.step import Step
from stepper.step_logging import (
    format_module_end,
    format_module_start,
    format_step_end,
    format_step_fail,
    format_step_start,
)

_LOGGER = logging.getLogger(__name__)


class Stage:
    """Base stage: declare `steps = (...)` (membership); run order comes from `depends()`."""

    stage_name: ClassVar[str] = ""             # inferred from the class name
    steps: ClassVar[tuple[Step[Any], ...]] = ()  # membership set — NOT run order (that's from deps)
    _scheduler: ClassVar[Scheduler]            # dependency runner, built per subclass

    def __init_subclass__(cls) -> None:
        cls._check_steps()
        cls.stage_name = cls.__name__.removesuffix("Stage")   # "ExtractStage" -> "Extract"

        for step in cls.steps:
            step.claim(cls)

        # After claim(), so cross-stage deps resolve; building the scheduler validates
        # the dep graph (unknown target / cycle) at class creation, before any run.
        cls._scheduler = Scheduler(cls.steps, label=cls.__name__)

    @classmethod
    def _check_steps(cls) -> None:
        """Reject a stage missing `steps`, listing a non-Step, or repeating a step
        (which would run and overwrite its own output)."""
        if not cls.steps or not all(isinstance(s, Step) for s in cls.steps):
            raise TypeError(f"{cls.__name__} must declare `steps = (...)` listing its steps.")
        if len(set(cls.steps)) != len(cls.steps):
            raise TypeError(f"{cls.__name__}.steps has a duplicate step.")

    def __init__(self, *, persist_service: PersistService, hooks: Hooks | None = None) -> None:
        """
        Args:
            persist_service: Backend each step fetches its inputs from and persists its
                output to.
            hooks: Wraps this stage and its steps (default: no-op). A Pipeline overrides
                this only when the Pipeline itself was given hooks.
        """
        self._persist = persist_service
        self._hooks: Hooks = hooks or NoOpHooks()
        # name -> runner, so run_step/get_steps can target one step by name
        self._runners: dict[str, Callable[[], Coroutine[Any, Any, Any]]] = {
            step.name: self._get_runner_for(step) for step in self.steps
        }

    def _get_runner_for(self, step: Step[Any]) -> Callable[[], Coroutine[Any, Any, Any]]:
        """Return a coroutine that runs the step: fetch inputs, run, persist, and log."""
        def get_step_key_for(s: Step[Any]) -> str:
            return "/".join((s.get_owner().stage_name, s.name))

        async def run() -> Any:
            deps = step.dependencies()
            optional = step.optional_dependencies()
            input_type = ", ".join(dep.model.__name__ if dep.model is not None else "None" for dep in deps.values()) or "None"
            output_type = step.model.__name__ if step.model is not None else "None"
            _LOGGER.info(format_step_start(step_name=step.name, input_type=input_type, output_type=output_type))
            started = perf_counter()
            try:
                with self._hooks.step(
                    stage_name=self.stage_name,
                    step_name=step.name,
                    input_type=input_type,
                    output_type=output_type,
                ) as report:
                    # Grab each declared input. A required dep with no persisted value
                    # raises (as always); an optional dep with none reads back as None.
                    inputs: dict[str, Any] = {}
                    for name, dep in deps.items():
                        try:
                            inputs[name] = self._persist.fetch(get_step_key_for(dep), dep.model)
                        except FileNotFoundError:
                            if name not in optional:
                                raise
                            inputs[name] = None

                    # Run the step
                    result = await step.fn(self, **inputs)

                    # Persist the result if the step declares an output model
                    if step.model is not None:
                        self._persist.persist(get_step_key_for(step), result, step.model)
                        # Hand the output to the hook's StepReport (if it yielded one).
                        if report is not None:
                            report.set_output(result)
            except Exception as exc:
                elapsed_ms = int((perf_counter() - started) * 1000)
                _LOGGER.exception(format_step_fail(step_name=step.name, elapsed_ms=elapsed_ms, error_type=type(exc).__name__))
                raise

            elapsed_ms = int((perf_counter() - started) * 1000)
            _LOGGER.info(format_step_end(step_name=step.name, elapsed_ms=elapsed_ms, output_type=type(result).__name__))
            return result

        return run

    def get_steps(self) -> list[str]:
        return list(self._runners)

    async def run_step(self, step_name: str) -> Any:
        runner = self._runners.get(step_name)
        if runner is None:
            raise ValueError(f"Unknown step: {step_name}")
        return await runner()

    async def run_steps(self, *, fail_fast: bool = False) -> list[Any]:
        """Run every step by its dependency DAG. `fail_fast=True` cancels in-flight steps
        and re-raises on the first failure; the default records it, skips its dependents,
        and lets independent branches finish."""
        _LOGGER.info(format_module_start(module_name=self.stage_name, step_count=len(self._runners)))
        started = perf_counter()
        try:
            with self._hooks.stage(stage_name=self.stage_name, step_count=len(self._runners)):
                # The scheduler owns the loop; we hand it run_step (how to run one by name).
                return await self._scheduler.run(self.run_step, fail_fast=fail_fast)
        finally:
            elapsed_ms = int((perf_counter() - started) * 1000)
            _LOGGER.info(format_module_end(module_name=self.stage_name, elapsed_ms=elapsed_ms))
