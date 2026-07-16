"""stepper — a tiny dependency-scheduled pipeline framework.

Declare steps with `@step`, wire inputs with `depends`, group them into a `Stage`,
and run a `Pipeline`. Values persist per step via a `PersistService`; run order is
derived from the dependency DAG, not declaration order.
"""

from stepper.hooks import Hooks, NoOpHooks, StepReport
from stepper.logging_config import configure_logging
from stepper.persist import DiskPersistService, PersistService
from stepper.pipeline import Pipeline, StageFactory
from stepper.scheduler import Scheduler
from stepper.stage import Stage
from stepper.step import Step, depends, step

__all__ = [
    "Pipeline",
    "StageFactory",
    "Stage",
    "Step",
    "step",
    "depends",
    "Scheduler",
    "PersistService",
    "DiskPersistService",
    "Hooks",
    "NoOpHooks",
    "StepReport",
    "configure_logging",
]
