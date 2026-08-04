"""stepper — a tiny pipeline framework built on one recursive node.

A `Node` is either a `Step` (a leaf: one async function, one persisted value) or a `Flow`
(a container of nodes, which are steps or other flows). There is nothing above a flow and
nothing below a step, so any flow you can build is a flow you can run:

    await MyFlow(run_id="run-1").run()
    await MyFlow(run_id="run-1").run("tiktok/publish/upload")

Settings — the persist service, `run_id`, hooks — are given to the flow you run and
inherited by everything beneath it. Constructing a flow constructs its subtree
with it, so a flow instance is a whole run's tree. Values persist per step, keyed by the
node's *path*, so mounting one flow twice gives two namespaces. Run order is derived from
the `depends()` DAG, unless a flow declares `edges`: a full control-flow graph from `START`
to `EXIT`, which is how it expresses a loop, checkpointing so a crashed run resumes
mid-graph.
"""

from stepper.edges import EXIT, START, Edge, edge
from stepper.flow import Flow, FlowRef
from stepper.hooks import Hooks, StepReport
from stepper.logging_config import configure_logging
from stepper.node import Node
from stepper.producer import Producer, Requirement, require
from stepper.persist import (
    DiskPersistService,
    InMemoryPersistService,
    Persistable,
    PersistService,
)
from stepper.step import Step, depends, optional_depends, step

__all__ = [
    "Node",
    "Flow",
    "FlowRef",
    "Producer",
    "Requirement",
    "require",
    "Step",
    "step",
    "depends",
    "optional_depends",
    "edge",
    "Edge",
    "START",
    "EXIT",
    "PersistService",
    "DiskPersistService",
    "InMemoryPersistService",
    "Persistable",
    "Hooks",
    "StepReport",
    "configure_logging",
]
