"""Control-flow edges: the other way a flow can declare its order.

Every flow declares its whole graph, and it runs sequentially: one node, then whichever
edge its result matches, until an edge reaches `EXIT`. Nothing is inferred. Every node the
flow declares must appear in the graph, `START` says where it begins, and `depends()` is
pure data wiring with no say in ordering — it is checked against the graph instead.

A *node* is a step or a nested flow, so an edge routes between flows exactly as it routes
between steps. Declaring the whole thing is what lets a flow loop, which `depends()` never
could:

    edges = (
        edge(START).to(edit),
        edge(edit).to(audit),
        edge(audit).when(lambda r: r.passed).to(EXIT).otherwise(analysis),
        edge(analysis)
            .when(lambda r: r.rounds >= 8).to(EXIT)
            .when(lambda r: r.remeasures < 2).to(audit)
            .otherwise(edit),
    )

Nodes stay exactly what they were: inputs via `depends()`, one output model. They know
nothing about routing — no marker types, no branch tokens in the return annotation, so
what lands on disk is still a real domain model. Declaring the edges *below* the steps is
what makes the back edge legal: `analysis -> edit` is a forward reference at that point in
the class body.

`edge(step)` is typed on that step's return model, so a predicate's only argument is the
value its source produced — `r.passed` above is checked against `AuditResult`, not `Any`.
The predicate is *called*, never inspected: nothing here reads a return type to decide a
route. Routing is only ever the edges you declared.

A branch sees its source's output and nothing else, on purpose. The node that just ran is
the thing reporting what happened, so whatever a decision needs belongs in its output
model — which also makes the persisted artifact a complete account of the state at that
point. A step needing a fact it doesn't own declares a `depends()` for it like any other
input.

**The framework counts nothing.** There is no round number, no loop state. Real loops
track several quantities at once — rounds, retries, budget raises — so a single
framework-owned counter would be one of them at best, and a second source of truth about
a loop that already records its own. Counters live in the step's output model, where
they're typed, persisted, and visible on disk; `r.rounds >= 8` above is a field the step
wrote. The one number the framework keeps is `Flow.max_steps`, a runaway fuse that is
never exposed to a predicate.

Predicates take the source's result and must be pure functions of it: they are
re-evaluated when a crashed run resumes, so no clock, no randomness, no network.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar, overload

if TYPE_CHECKING:
    from stepper.step import Step

R = TypeVar("R")

# One member of a flow: the name it's mounted under, and the declaration object itself
# (a `Step` handle or an unbound `Flow`). Membership is by *identity* — two flows may each
# own a step called "validate", and an edge naming the wrong one must be rejected, not
# quietly rebound to the local one.
Member = tuple[str, Any]


class _Start:
    """Sentinel edge source: where the graph begins. `edge(START).to(first_node)`."""

    __slots__ = ()
    name = "START"

    def __repr__(self) -> str:
        return "START"


class _Exit:
    """Sentinel edge target: the graph is finished and the flow is done."""

    __slots__ = ()
    name = "EXIT"

    def __repr__(self) -> str:
        return "EXIT"


START = _Start()
EXIT = _Exit()


class Edge(Generic[R]):
    """One source's outgoing control flow, built by `edge(source)`.

    `R` is the source node's output model, so every predicate's argument is typed:
    `edge(audit).when(lambda r: r.passed)` checks `passed` against `AuditResult`.

    Either unconditional (`.to(target)`) or a branch chain (`.when(pred).to(target)` one
    or more times, closed by `.otherwise(target)`). The chain is ordered — the first
    predicate that returns True wins — and `otherwise` is mandatory on a branch chain so
    exhaustiveness is structural rather than something we hope the lambdas cover.
    """

    __slots__ = ("source", "_branches", "_default", "_pending", "_unconditional")

    def __init__(self, source: Any) -> None:
        self.source = source
        self._branches: list[tuple[Callable[[R], bool], Any]] = []
        self._default: Any | None = None
        self._pending: Callable[[R], bool] | None = None
        self._unconditional = False

    def when(self, predicate: Callable[[R], bool]) -> "Edge[R]":
        """Take this branch when `predicate(result)` is True — `result` being whatever the
        source node just returned."""
        if self._unconditional:
            raise TypeError(f"edge({_label(self.source)}): .when(...) after an unconditional .to(...).")
        if self._pending is not None:
            raise TypeError(f"edge({_label(self.source)}): .when(...) twice with no .to(...) between them.")
        if self._default is not None:
            raise TypeError(f"edge({_label(self.source)}): .when(...) after .otherwise(...).")
        self._pending = predicate
        return self

    def to(self, target: Any) -> "Edge[R]":
        """Target of the pending `.when(...)`, or — with no pending predicate — this
        source's single unconditional target."""
        if self._pending is None:
            if self._branches or self._default is not None:
                raise TypeError(f"edge({_label(self.source)}): unconditional .to(...) mixed with branches; use .otherwise(...).")
            self._unconditional = True
            self._default = target
        else:
            self._branches.append((self._pending, target))
            self._pending = None
        return self

    def otherwise(self, target: Any) -> "Edge[R]":
        """Fallback when no `.when(...)` matched. Required on any branching edge."""
        if self._unconditional:
            raise TypeError(f"edge({_label(self.source)}): .otherwise(...) on an unconditional edge.")
        if not self._branches:
            raise TypeError(f"edge({_label(self.source)}): .otherwise(...) with no .when(...) before it.")
        if self._default is not None:
            raise TypeError(f"edge({_label(self.source)}): .otherwise(...) declared twice.")
        self._default = target
        return self

    @property
    def is_branching(self) -> bool:
        return bool(self._branches)

    def targets(self) -> list[Any]:
        return [t for _, t in self._branches] + ([self._default] if self._default is not None else [])

    def check(self) -> None:
        """Reject a half-built edge — raised while the flow is being declared."""
        if self._pending is not None:
            raise TypeError(f"edge({_label(self.source)}): .when(...) with no .to(...) after it.")
        if self._default is None:
            if not self._branches:
                raise TypeError(f"edge({_label(self.source)}): declared with no .to(...) — it routes nowhere.")
            raise TypeError(
                f"edge({_label(self.source)}): branching edge needs .otherwise(...) so every result routes somewhere."
            )

    def resolve(self, result: R) -> Any:
        """Where control goes after the source node returned `result`: the first branch
        whose predicate matches, else the default."""
        for predicate, target in self._branches:
            if predicate(result):
                return target
        assert self._default is not None      # check() ran when the flow was declared
        return self._default


def _label(obj: Any) -> str:
    """Best-effort name for an error message, before membership is resolved."""
    return getattr(obj, "name", None) or type(obj).__name__


@overload
def edge(source: "Step[R]") -> Edge[R]: ...
@overload
def edge(source: _Start) -> Edge[Any]: ...
@overload
def edge(source: Any) -> Edge[Any]: ...
def edge(source: Any) -> Edge[Any]:
    """Start declaring a source's outgoing control flow — a `Step`, a nested `Flow`, or
    `START` for the edge into the graph's first node. The returned `Edge` is typed on a
    step's output, so predicates get a typed `result`. See the module docstring."""
    return Edge(source)


class Graph:
    """A flow's validated control-flow graph: where it starts and the edge out of every
    node. Built once when the flow is declared, so a malformed graph raises then — same
    contract as the dependency DAG it replaces.

    "Declare the whole graph" is enforced here: every node the flow lists must have an
    edge out and be reachable from `START`, and `EXIT` must be reachable. Nothing is
    inferred and nothing is optional. On top of that, every node's required `depends()`
    must be *guaranteed* to have run before it — see `_check_deps_available`.

    Nodes are identified by their member name; `required`/`optional` are the flow's
    already-resolved dependency maps (`{node: {node it needs}}` and `[(consumer,
    producer)]`), so nothing here has to know what a step or a flow is.
    """

    def __init__(
        self,
        edges: tuple[Edge[Any], ...],
        members: Sequence[Member],
        label: str,
        required: Mapping[str, set[str]],
        optional: Iterable[tuple[str, str]],
    ) -> None:
        self._members = list(members)                     # keep alive: ids are the identity
        self._name_by_id = {id(obj): name for name, obj in self._members}
        self._by_source: dict[str, Edge[Any]] = {}
        where = f"{label} " if label else ""
        names = {name for name, _ in self._members}
        entry: str | None = None

        for e in edges:
            e.check()
            if isinstance(e.source, _Start):
                if entry is not None:
                    raise TypeError(f"{where}edges: more than one edge(START); the graph has one way in.")
                if e.is_branching:
                    raise TypeError(f"{where}edges: edge(START) must be unconditional — there is no result to branch on.")
                target = e.targets()[0]
                if isinstance(target, _Exit):
                    raise TypeError(f"{where}edges: edge(START).to(EXIT) runs nothing.")
                entry = self._name_of(target)
                if entry is None:
                    raise TypeError(f"{where}edges: edge(START) routes to {_label(target)!r}, which is not a node of this flow.")
                continue
            source = self._name_of(e.source)
            if source is None:
                raise TypeError(f"{where}edges: {_label(e.source)!r} is not a node of this flow.")
            if source in self._by_source:
                raise TypeError(f"{where}edges: {source!r} has more than one edge(...); one per node.")
            self._by_source[source] = e

        if entry is None:
            raise TypeError(f"{where}edges: no edge(START).to(...), so nothing says where the graph begins.")
        self.entry: str = entry

        for source, e in self._by_source.items():
            for t in e.targets():
                if not isinstance(t, _Exit) and self._name_of(t) is None:
                    raise TypeError(f"{where}edges: {source!r} routes to {_label(t)!r}, which is not a node of this flow.")

        missing = sorted(names - set(self._by_source))
        if missing:
            raise TypeError(
                f"{where}edges: {', '.join(missing)} declare no edge(...) out. A flow with edges "
                f"declares its whole graph — every node needs one."
            )

        self._check_reachability(where, names)
        self._check_deps_available(where, names, required, optional)

    def _name_of(self, obj: Any) -> str | None:
        return self._name_by_id.get(id(obj))

    def _targets_of(self, name: str) -> list[str]:
        """Names of the nodes `name` can route to (EXIT dropped)."""
        return [
            n for n in (self._name_of(t) for t in self._by_source[name].targets() if not isinstance(t, _Exit))
            if n is not None
        ]

    def _predecessors(self, names: Iterable[str]) -> dict[str, set[str]]:
        preds: dict[str, set[str]] = {n: set() for n in names}
        for source in self._by_source:
            for t in self._targets_of(source):
                preds[t].add(source)
        return preds

    def _check_reachability(self, where: str, names: set[str]) -> None:
        """Every node must be reachable from the entry, and every node must be able to
        reach `EXIT` — otherwise the graph either contains dead code or contains a branch
        that, once taken, can never finish.

        The second half is per-node, not "somewhere in the graph there's an EXIT". A
        branch whose arm drops into a sub-cycle with no way out is exactly as broken as a
        graph with no EXIT at all; it just takes a run to find out.
        """
        seen: set[str] = set()
        frontier = [self.entry]
        while frontier:
            name = frontier.pop()
            if name in seen:
                continue
            seen.add(name)
            frontier.extend(self._targets_of(name))

        unreachable = sorted(names - seen)
        if unreachable:
            raise TypeError(f"{where}edges: {', '.join(unreachable)} is unreachable from START.")

        # Walk backwards from the nodes that route to EXIT directly.
        can_exit = {
            n for n in names if any(isinstance(t, _Exit) for t in self._by_source[n].targets())
        }
        changed = True
        while changed:
            changed = False
            for name in names - can_exit:
                if any(t in can_exit for t in self._targets_of(name)):
                    can_exit.add(name)
                    changed = True

        stuck = sorted(names - can_exit)
        if stuck:
            raise TypeError(
                f"{where}edges: {', '.join(stuck)} has no route to EXIT — once control reaches "
                f"{'it' if len(stuck) == 1 else 'them'}, the graph could never finish."
            )

    def _reaches(self, source: str, target: str) -> bool:
        """Whether control can get from `source` to `target` by following edges — i.e.
        whether *some* execution runs `source` before `target`."""
        seen: set[str] = set()
        frontier = self._targets_of(source)
        while frontier:
            name = frontier.pop()
            if name == target:
                return True
            if name in seen:
                continue
            seen.add(name)
            frontier.extend(self._targets_of(name))
        return False

    def _check_deps_available(
        self,
        where: str,
        names: set[str],
        required: Mapping[str, set[str]],
        optional: Iterable[tuple[str, str]],
    ) -> None:
        """A node's in-flow `depends()` has to line up with the graph, and the bar differs
        by kind:

        - **required** — the producer must have run on *every* path from START. A step
          fetches its inputs off the backend, so a producer that hasn't run has nothing
          stored and the fetch raises. `edge(START).to(b)` where `b = depends(a)` is
          impossible to run, not merely unusual.
        - **optional** — the producer must be able to run before the consumer on *at least
          one* path. `optional_depends` exists so a first pass can read None, not so a
          parameter can be None forever; if no route from producer to consumer exists, the
          wiring is dead and the graph doesn't mean what it looks like it means.

        The required side is a "must have executed" fixpoint: what's guaranteed at a node
        is the intersection over its predecessors of (guaranteed at that predecessor, plus
        the predecessor itself). The entry gets the empty set — START guarantees nothing —
        which is what makes a first-pass back-edge read come out unavailable, and why it
        has to be optional. The optional side is plain reachability, which is why a back
        edge passes: in `START -> b`, `b -> a`, `a -> b`, the second visit to `b` does have
        `a` behind it. Deps resolved outside this flow are excluded from both: those are
        persisted inputs from elsewhere in the tree, not this graph's business.
        """
        ordered = sorted(names)
        preds = self._predecessors(ordered)
        available: dict[str, set[str]] = {n: set(ordered) for n in ordered}
        available[self.entry] = set()

        changed = True
        while changed:
            changed = False
            for name in ordered:
                if name == self.entry:
                    continue          # START contributes nothing, so the entry stays empty
                fresh = set.intersection(*(available[p] | {p} for p in preds[name]))
                if fresh != available[name]:
                    available[name] = fresh
                    changed = True

        for name in ordered:
            unmet = sorted(required.get(name, set()) - available[name])
            if unmet:
                raise TypeError(
                    f"{where}edges: {name!r} requires {', '.join(repr(u) for u in unmet)}, which "
                    f"{'is' if len(unmet) == 1 else 'are'} not guaranteed to have run before it on every "
                    f"path from START — the fetch would find nothing. Route through it first, or wire "
                    f"the parameter with optional_depends(...) if None is a valid first pass."
                )

        for consumer, producer in sorted(set(optional)):
            if not self._reaches(producer, consumer):
                raise TypeError(
                    f"{where}edges: {consumer!r} optionally depends on {producer!r}, but no path "
                    f"leads from {producer!r} back to {consumer!r} — the value would be None on "
                    f"every pass. Drop the dependency, or route so {producer!r} can precede it."
                )

    def terminals(self) -> list[str]:
        """Nodes with an edge to `EXIT` — the ones a run can end on, and so the ones whose
        value can be this flow's. Every one of them has to produce the same model."""
        return [
            name for name, e in self._by_source.items()
            if any(isinstance(t, _Exit) for t in e.targets())
        ]

    def edge_for(self, name: str) -> Edge[Any]:
        return self._by_source[name]

    def target_name(self, obj: Any) -> str | None:
        """Member name of an edge target, or None for `EXIT`."""
        return None if isinstance(obj, _Exit) else self._name_of(obj)


