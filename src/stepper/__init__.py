"""stepper — a tiny dependency-scheduled pipeline framework.

Declare steps with `@step`, wire inputs with `depends`, group them into a `Stage`,
and run a `Pipeline`. Values persist per step via a `PersistService`; run order is
derived from the dependency DAG, not declaration order. A stage can instead declare
`edges` — a full control-flow graph from `START` to `EXIT`, which is how it expresses a
loop; it runs sequentially and checkpoints progress so a crashed run resumes mid-graph.
"""

from stepper.edges import EXIT, START, Edge, edge
from stepper.hooks import Hooks, NoOpHooks, StepReport
from stepper.logging_config import configure_logging
from stepper.persist import (
    DiskPersistService,
    InMemoryPersistService,
    Persistable,
    PersistService,
)
from stepper.pipeline import Pipeline, StageFactory
from stepper.scheduler import Scheduler
from stepper.stage import Stage
from stepper.step import Step, depends, optional_depends, step

__all__ = [
    "Pipeline",
    "StageFactory",
    "Stage",
    "Step",
    "step",
    "depends",
    "optional_depends",
    "edge",
    "Edge",
    "START",
    "EXIT",
    "Scheduler",
    "PersistService",
    "DiskPersistService",
    "InMemoryPersistService",
    "Persistable",
    "Hooks",
    "NoOpHooks",
    "StepReport",
    "configure_logging",
]
