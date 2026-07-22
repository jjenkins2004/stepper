"""Pipeline — one named, run-scoped set of stages.

Owns everything internal to running a single pipeline: its name, the PersistService
(namespaced by `output_root/name`, plus `/run_id` when given so runs never collide),
building and running
its stages, and listing its steps. This is the stuff that doesn't change run to run —
the pipeline's shape.

Where output lands is configurable: pass `output_root` (relative paths resolve against
cwd) or hand in your own `persist_service` for a non-disk backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from stepper.hooks import Hooks
from stepper.persist import DiskPersistService, PersistService
from stepper.stage import Stage

# Builds a stage from the shared PersistService. Per-run inputs (e.g. a
# spreadsheet path) are baked into the closure by the pipeline's wiring.
StageFactory = Callable[[PersistService], Stage]


class Pipeline:
    """A pipeline as data: its name and its stages, in run order. Iterating the
    stages runs the pipeline start to finish."""

    def __init__(
        self,
        *,
        name: str,
        stages: dict[str, StageFactory],
        run_id: str | None = None,
        output_root: Path | str = Path("output"),
        persist_service: PersistService | None = None,
        hooks: Hooks | None = None,
        fail_fast: bool = False,
    ) -> None:
        """
        Args:
            name: Pipeline name; the first path segment of the default output dir
                (e.g. "orders" -> output/orders/...).
            stages: Maps stage name -> a factory that builds the stage from the shared
                PersistService. Dict order is run order when you run `module="all"`.
            run_id: Optional per-run label. Adds a `/run_id` subdir so repeat runs don't
                overwrite each other. Only the default disk backend uses it — a custom
                `persist_service` sets its own run_id.
            output_root: Root dir for the default disk backend. Final path is
                `output_root/name[/run_id]`. Relative paths resolve against cwd.
            persist_service: Bring-your-own backend (in-memory, object store, DB). When
                given it wins — `output_root`/`run_id` are ignored and no disk backend
                is built.
            hooks: Wraps every stage and step this pipeline runs (tracing/metrics/etc.).
                Omit and each stage keeps whatever hooks it was built with.
            fail_fast: When True, a stage cancels its in-flight steps and re-raises on the
                first step failure. Default False: record the failure, skip its dependents,
                and let independent branches finish. Applies to every stage this pipeline
                runs; a single-step run always re-raises regardless.
        """
        self.name = name
        self.run_id = run_id
        self._fail_fast = fail_fast
        self._stage_factories = stages
        # None means "pipeline sets no hooks" — each stage keeps its own (see build_stage).
        self._hooks: Hooks | None = hooks
        # Explicit persist_service wins; otherwise a disk backend rooted at
        # output_root/name, handed run_id so it lands per-run output under it.
        self.persist_service: PersistService = persist_service or DiskPersistService(
            base_dir=Path(output_root) / name, run_id=run_id
        )

    def stage_names(self) -> list[str]:
        return list(self._stage_factories)

    def build_stage(self, stage_name: str) -> Stage:
        try:
            factory = self._stage_factories[stage_name]
        except KeyError:
            raise ValueError(
                f"Pipeline '{self.name}' has no stage '{stage_name}'. "
                f"Available: {self.stage_names()}"
            ) from None
        stage = factory(self.persist_service)
        # Only override when the pipeline was given hooks; otherwise the stage keeps its own.
        if self._hooks is not None:
            stage._hooks = self._hooks  # pipeline-level hooks wrap every stage/step it runs
        return stage

    def step_map(self) -> dict[str, list[str]]:
        """Return ``{stage_name: [step_names]}`` (e.g. for a menu). Builds throwaway
        stages just to read their step names; safe before a run does any work."""
        return {name: self.build_stage(name).get_steps() for name in self.stage_names()}

    def _resolve_stage_names(self, module: str, step: str | None) -> tuple[str, ...]:
        if module == "all":
            if step:
                raise ValueError("step is not valid with module 'all'")
            return tuple(self.stage_names())
        return (module,)

    async def run(self, *, module: str, step: str | None = None) -> Any:
        """Run the whole pipeline (``module="all"``), one stage, or one step, and return
        the last thing run — so a caller can read a pipeline's final output without going
        back to the PersistService. A single-step run returns that step's value; a stage or
        ``"all"`` run returns the final stage's step results (a list, in declaration order,
        from `Stage.run_steps`)."""
        result: Any = None
        for name in self._resolve_stage_names(module, step):
            stage = self.build_stage(name)
            if step and name == module:
                result = await stage.run_step(step)
            else:
                result = await stage.run_steps(fail_fast=self._fail_fast)
        return result
