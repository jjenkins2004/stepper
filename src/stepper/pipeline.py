"""Pipeline — one named, run-scoped set of stages.

Owns everything internal to running a single pipeline: its name, the PersistService
(namespaced by `output_root/name/run_id` so runs never collide), building and running
its stages, and listing its steps. This is the stuff that doesn't change run to run —
the pipeline's shape.

Where output lands is configurable: pass `output_root` (relative paths resolve against
cwd) or hand in your own `persist_service` for a non-disk backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from stepper.hooks import Hooks, NoOpHooks
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
        run_id: str,
        stages: dict[str, StageFactory],
        output_root: Path | str = Path("output"),
        persist_service: PersistService | None = None,
        hooks: Hooks | None = None,
    ) -> None:
        self.name = name
        self.run_id = run_id
        self._stage_factories = stages
        self._hooks: Hooks = hooks or NoOpHooks()
        # Explicit persist_service wins; otherwise a disk backend under output_root/name/run_id.
        self.persist_service: PersistService = persist_service or DiskPersistService(
            base_dir=Path(output_root) / name / run_id
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

    async def run(self, *, module: str, step: str | None = None) -> None:
        """Run the whole pipeline (``module="all"``), one stage, or one step."""
        for name in self._resolve_stage_names(module, step):
            stage = self.build_stage(name)
            if step and name == module:
                await stage.run_step(step)
            else:
                await stage.run_steps()
