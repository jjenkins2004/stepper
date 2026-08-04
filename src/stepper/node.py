"""Node: the one interface. A step and a flow are the same kind of thing to their parent.

A `Step` is a leaf — an async function with one persisted value. A `Flow` is a container
of nodes, which are steps or other flows. Nothing else exists: there is no pipeline layer
above and no special case below, so any flow you can build is a flow you can run.

Three things are uniform across both:

**Path.** A node's path is the `/`-joined chain of names from the root down, root included
(`"push/transform/strip"`) — the attribute name each node was declared under, all the way
up. It is also the key the node persists under, so where a value lands is decided by where
the node sits, not by what class it came from.

**Address.** `run(target)` takes a path. Each node eats the first segment and hands the
rest down, so `root.run("tiktok/publish/upload")` is `tiktok` asking `publish` to run
`upload`. A step is the end of the line — give one a target and it says so.

**Context.** One `Context` is built when the root is mounted and shared by every node in
the run: the persist service, the hooks, and the `run_id` that prefixes keys.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from stepper.hooks import Hooks
from stepper.persist import PersistService

if TYPE_CHECKING:
    from stepper.flow import Flow


def join(prefix: str, name: str) -> str:
    """Extend a node path."""
    return f"{prefix}/{name}"


def split(target: str) -> tuple[str, str]:
    """Split an address into `(first segment, remainder)`. The remainder is `""` at the
    end of the line."""
    head, _, rest = target.strip("/").partition("/")
    return head, rest


@dataclass(frozen=True)
class Context:
    """The run itself: what the root was given, shared by every node in the tree."""

    persist: PersistService
    hooks: tuple[Hooks, ...]
    run_id: str | None = None

    def key(self, path: str) -> str:
        """The persist key for a node path. `run_id` prefixes keys rather than paths, so
        two runs never collide on the backend while addresses stay run-independent —
        `run("tiktok/publish")` is the same string whichever run you're in."""
        return join(self.run_id, path) if self.run_id else path


class Node(ABC):
    """A step or a flow. Everything a parent needs from either is here."""

    name: str

    def __init__(self) -> None:
        self._path: str = ""
        self._ctx: Context | None = None
        self._parent: "Flow | None" = None

    @property
    def path(self) -> str:
        """Where this node is mounted, from the root down — and, for a step, the key it
        persists under. Addresses passed to `run()` are relative to the node you call it
        on, so they never repeat that node's own path."""
        return self._path

    @property
    def is_bound(self) -> bool:
        return self._ctx is not None

    @property
    def ctx(self) -> Context:
        if self._ctx is None:
            raise RuntimeError(
                f"{self.name} is not mounted. Run a flow directly and it mounts itself; a node "
                f"only becomes runnable once some flow mounts it."
            )
        return self._ctx

    @abstractmethod
    async def run(self, target: str | None = None, *, follow_edges: bool = False) -> Any:
        """Run this node, or the node at `target` beneath it. Same signature for a step and
        a flow — that sameness is what makes the tree recursive rather than layered."""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """This node as plain data, for printing or a menu."""
