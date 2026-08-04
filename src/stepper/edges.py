"""Control-flow edges: the other way a flow can declare its order.

Every flow declares its whole graph, and it runs that: one node, then whichever edge its
result matches, until an edge reaches `EXIT`. Nothing is inferred. Every node the
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

Several targets on one `.to(...)` is a **fan-out**: every arm runs, concurrently, and
control resumes at the node where they meet. Several *sources* is the shorthand for coming
back in — one edge declared once per source, which is what the arms need anyway:

    edges = (
        edge(START).to(plan),
        edge(plan).to(images, copy, pricing),          # three arms, at once
        edge(images, copy, pricing).to(assemble),      # and back together
        edge(assemble).to(EXIT),
    )

A multi-source edge routes unconditionally: its sources have no one result to branch on,
and no one `R` to type a predicate against. It is sugar and nothing else — `edge(images)`
elsewhere in the same tuple is the same "one edge per node" error it has always been.

The meeting point is not declared — it is derived, as the first node every arm is
guaranteed to reach, so a fan-out is one edge and nothing else. That derivation is what
makes the awkward cases fall out for free: an arm may branch, loop, or fan out again, and
`.when(...).to(a, b)` fans out on that branch alone. What it will not do is invent a join
that isn't there: arms whose only meeting point is `EXIT` are rejected, because a flow's
value is whatever node it ended on and several arms ending at once names no node.

Arms are independent by construction — disjoint, and enterable only through their fan-out —
which is checked when the class is created, along with the rest of the graph. A `depends()`
across two arms is rejected for the same reason: they run at the same time, so it is a race.
The join is where arms are read, and it may read all of them: every arm ran, so every arm's
output is behind it.

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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar, overload

if TYPE_CHECKING:
    from stepper.step import Step

R = TypeVar("R")

# Sentinel name for EXIT inside the internal graph analyses, and the separator used to name
# the virtual node standing in for one outcome's fan-out. Neither can collide with a member
# name: `#` is not legal in an attribute name and EXIT is a member of nothing.
_EXIT_NAME = "EXIT"
_FAN_SEP = "#"

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
    """The outgoing control flow of a node — or, given several, of each of them.

    `R` is the source node's output model, so every predicate's argument is typed:
    `edge(audit).when(lambda r: r.passed)` checks `passed` against `AuditResult`.

    Either unconditional (`.to(target)`) or a branch chain (`.when(pred).to(target)` one
    or more times, closed by `.otherwise(target)`). The chain is ordered — the first
    predicate that returns True wins — and `otherwise` is mandatory on a branch chain so
    exhaustiveness is structural rather than something we hope the lambdas cover.

    `edge(a, b, c).to(join)` declares that same edge once for each source — the fan-in
    counterpart to `.to(a, b, c)`, and nothing more than shorthand for three edges. It
    routes unconditionally: several sources have no one result to branch on, and no one `R`
    to type a predicate against.
    """

    __slots__ = ("sources", "_branches", "_default", "_pending", "_unconditional")

    def __init__(self, *sources: Any) -> None:
        if not sources:
            raise TypeError("edge(): needs at least one source node.")
        self.sources = sources
        self._branches: list[tuple[Callable[[R], bool], tuple[Any, ...]]] = []
        self._default: tuple[Any, ...] | None = None
        self._pending: Callable[[R], bool] | None = None
        self._unconditional = False

    @property
    def source(self) -> Any:
        """The single source, for the usual one-source edge."""
        return self.sources[0]

    @property
    def is_fan_in(self) -> bool:
        return len(self.sources) > 1

    def _me(self) -> str:
        return f"edge({', '.join(_label(s) for s in self.sources)})"

    def when(self, predicate: Callable[[R], bool]) -> "Edge[R]":
        """Take this branch when `predicate(result)` is True — `result` being whatever the
        source node just returned."""
        if self.is_fan_in:
            raise TypeError(
                f"{self._me()}: a multi-source edge routes unconditionally — its sources have "
                f"no one result to branch on. Declare each source's own edge(...) to branch."
            )
        if self._unconditional:
            raise TypeError(f"{self._me()}: .when(...) after an unconditional .to(...).")
        if self._pending is not None:
            raise TypeError(f"{self._me()}: .when(...) twice with no .to(...) between them.")
        if self._default is not None:
            raise TypeError(f"{self._me()}: .when(...) after .otherwise(...).")
        self._pending = predicate
        return self

    def to(self, *targets: Any) -> "Edge[R]":
        """Target of the pending `.when(...)`, or — with no pending predicate — this
        source's single unconditional target.

        Several targets is a **fan-out**: they all run, concurrently, and control resumes at
        the node where their arms meet. See the module docstring."""
        if not targets:
            raise TypeError(f"{self._me()}: .to(...) needs at least one target.")
        if self.is_fan_in and len(targets) > 1:
            raise TypeError(
                f"{self._me()}: a multi-source edge fans in, so it can't also fan out. Give each "
                f"source its own edge(...) to fan out from."
            )
        if self._pending is None:
            if self._branches or self._default is not None:
                raise TypeError(f"{self._me()}: unconditional .to(...) mixed with branches; use .otherwise(...).")
            self._unconditional = True
            self._default = tuple(targets)
        else:
            self._branches.append((self._pending, tuple(targets)))
            self._pending = None
        return self

    def otherwise(self, *targets: Any) -> "Edge[R]":
        """Fallback when no `.when(...)` matched. Required on any branching edge. Several
        targets fan out, exactly as on `.to(...)`."""
        if self._unconditional:
            raise TypeError(f"{self._me()}: .otherwise(...) on an unconditional edge.")
        if not self._branches:
            raise TypeError(f"{self._me()}: .otherwise(...) with no .when(...) before it.")
        if self._default is not None:
            raise TypeError(f"{self._me()}: .otherwise(...) declared twice.")
        if not targets:
            raise TypeError(f"{self._me()}: .otherwise(...) needs at least one target.")
        self._default = tuple(targets)
        return self

    @property
    def is_branching(self) -> bool:
        return bool(self._branches)

    def outcomes(self) -> list[tuple[Any, ...]]:
        """One entry per possible route out, in the order they're tried: each `.when(...)`
        arm, then the default. An entry with more than one target is a fan-out."""
        return [t for _, t in self._branches] + ([self._default] if self._default is not None else [])

    def targets(self) -> list[Any]:
        """Every target this edge can route to, fan-out arms flattened in."""
        return [t for outcome in self.outcomes() for t in outcome]

    def check(self) -> None:
        """Reject a half-built edge — raised while the flow is being declared."""
        if self._pending is not None:
            raise TypeError(f"{self._me()}: .when(...) with no .to(...) after it.")
        if self._default is None:
            if not self._branches:
                raise TypeError(f"{self._me()}: declared with no .to(...) — it routes nowhere.")
            raise TypeError(
                f"{self._me()}: branching edge needs .otherwise(...) so every result routes somewhere."
            )

    def resolve_index(self, result: R) -> int:
        """Which `outcomes()` entry `result` routes to: the first branch whose predicate
        matches, else the default (the last entry)."""
        for i, (predicate, _) in enumerate(self._branches):
            if predicate(result):
                return i
        return len(self._branches)

    def resolve(self, result: R) -> Any:
        """Where control goes after the source node returned `result`, when that's a single
        place. A fan-out has no one target — `Graph.route` is what the runner asks."""
        outcome = list(self.outcomes()[self.resolve_index(result)])
        if len(outcome) > 1:
            raise TypeError(
                f"{self._me()}: that result fans out to {len(outcome)} nodes; it has no single target."
            )
        return outcome[0]


def _label(obj: Any) -> str:
    """Best-effort name for an error message, before membership is resolved."""
    return getattr(obj, "name", None) or type(obj).__name__


def _post_dominators(succ: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """Each node's *immediate post-dominator*: the nearest node every path out of it has to
    pass through on the way to EXIT.

    That is the question a fan-out asks — "where do these arms meet?" — and the textbook
    fixpoint answers it while handling the cases a hand-rolled walk would get wrong: a
    branch inside an arm (the join is past both halves), a loop inside an arm (a cycle that
    can exit still post-dominates fine), and fan-outs nested in fan-outs.

    `succ` maps every node to its successors, with `EXIT` as the sink; callers guarantee
    every node can reach it, so the post-dominator sets are never empty.
    """
    nodes = set(succ) | {_EXIT_NAME}
    pd: dict[str, set[str]] = {n: set(nodes) for n in nodes}
    pd[_EXIT_NAME] = {_EXIT_NAME}
    changed = True
    while changed:
        changed = False
        for name, targets in succ.items():
            fresh = {name} | set.intersection(*(pd[t] for t in targets))
            if fresh != pd[name]:
                pd[name] = fresh
                changed = True
    # A node's post-dominators form a chain, so the nearest one is the one with the most
    # post-dominators of its own.
    return {
        name: max(pd[name] - {name}, key=lambda c: len(pd[c]))
        for name in succ
    }


@overload
def edge(source: "Step[R]", /) -> Edge[R]: ...
@overload
def edge(source: _Start, /) -> Edge[Any]: ...
@overload
def edge(*sources: Any) -> Edge[Any]: ...
def edge(*sources: Any) -> Edge[Any]:
    """Start declaring a node's outgoing control flow — a `Step`, a nested `Flow`, or
    `START` for the edge into the graph's first node. The returned `Edge` is typed on a
    step's output, so predicates get a typed `result`.

    Name several sources and the edge is declared once for each of them, which is how a
    fan-out comes back in: `edge(images, copy, pricing).to(assemble)` is exactly the three
    edges it looks like. See the module docstring."""
    return Edge(*sources)


@dataclass(frozen=True)
class Fan:
    """One outcome that routes to several nodes: they run concurrently and control resumes
    at `join`, the single node where every arm ends up.

    `join` is *derived*, not declared — it is the immediate post-dominator of the fan-out,
    i.e. the first node every arm is guaranteed to reach. `regions[i]` is the set of nodes
    arm `i` owns on the way there, join excluded. The arms are checked to be disjoint and
    closed (nothing outside routes into one), which is what makes the region a real
    sub-graph the runner can hand to one concurrent walk.
    """

    source: str
    outcome: int
    arms: tuple[str, ...]
    join: str
    regions: tuple[frozenset[str], ...]

    @property
    def nodes(self) -> frozenset[str]:
        """Every node inside this fan-out, across all arms."""
        return frozenset().union(*self.regions)


@dataclass(frozen=True)
class Route:
    """Where a node's result sends control: one `target` (None being `EXIT`), or a `fan`
    whose arms run concurrently before control resumes at its join."""

    target: str | None
    fan: Fan | None


@dataclass(frozen=True)
class _Scope:
    """The one fan-out arm a node lives in — the nodes that arm owns, the fan's source, and
    the arm's entry. `None` instead of a `_Scope` means the node is at the flow's top level.
    Availability is analysed per scope, because arms run independently."""

    nodes: frozenset[str]
    source: str
    entry: str


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
            if any(isinstance(s, _Start) for s in e.sources):
                if e.is_fan_in:
                    raise TypeError(f"{where}edges: edge(START) names START alone; the graph has one way in.")
                if entry is not None:
                    raise TypeError(f"{where}edges: more than one edge(START); the graph has one way in.")
                if e.is_branching:
                    raise TypeError(f"{where}edges: edge(START) must be unconditional — there is no result to branch on.")
                if len(e.targets()) > 1:
                    raise TypeError(
                        f"{where}edges: edge(START) has one target — the graph has one way in. "
                        f"Fan out from the first node instead."
                    )
                target = e.targets()[0]
                if isinstance(target, _Exit):
                    raise TypeError(f"{where}edges: edge(START).to(EXIT) runs nothing.")
                entry = self._name_of(target)
                if entry is None:
                    raise TypeError(f"{where}edges: edge(START) routes to {_label(target)!r}, which is not a node of this flow.")
                continue
            # A multi-source edge is that edge declared once per source, so the "one edge
            # per node" rule below is what catches naming a node twice across the tuple.
            for obj in e.sources:
                source = self._name_of(obj)
                if source is None:
                    raise TypeError(f"{where}edges: {_label(obj)!r} is not a node of this flow.")
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
        self._preds = self._predecessors(sorted(names))
        self._fans: dict[str, Fan] = {}
        self._fans_by_join: dict[str, list[Fan]] = {}
        self._scope: dict[str, _Scope] = {}
        self._build_fans(where)
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

    # --- fan-out ---------------------------------------------------------------------

    def _build_fans(self, where: str) -> None:
        """Find every fan-out, work out where its arms meet, and check the region is a real
        one.

        A fan-out is *structured*: source, some arms, one join. The join is not declared —
        it is the immediate post-dominator of the fan-out, the first node every arm has to
        pass through. That's the only definition that agrees with `.when(...)`, loops inside
        an arm, and fan-outs nested in fan-outs, because post-dominance already accounts for
        all three.

        Post-dominance is computed on an *augmented* graph, where each multi-target outcome
        becomes a virtual node between the source and its arms. Without it a node that both
        fans out on one branch and routes single-file on another would have one join for
        both, which is wrong — the fan-out belongs to the branch, not the node.
        """
        succ: dict[str, list[str]] = {}
        arms_of: dict[str, tuple[str, ...]] = {}

        for source, e in self._by_source.items():
            outs: list[str] = []
            for idx, outcome in enumerate(e.outcomes()):
                if len(outcome) == 1:
                    t = outcome[0]
                    outs.append(_EXIT_NAME if isinstance(t, _Exit) else str(self._name_of(t)))
                    continue
                entries: list[str] = []
                for t in outcome:
                    if isinstance(t, _Exit):
                        raise TypeError(
                            f"{where}edges: {source!r} fans out to EXIT. A fan-out's arms all run, "
                            f"so they have to meet at a node — that node is what ends the flow."
                        )
                    entries.append(str(self._name_of(t)))
                if len(set(entries)) != len(entries):
                    raise TypeError(
                        f"{where}edges: {source!r} fans out to the same node twice; each arm of a "
                        f"fan-out is a distinct node."
                    )
                key = f"{source}{_FAN_SEP}{idx}"
                arms_of[key] = tuple(entries)
                outs.append(key)
            succ[source] = outs
        for key, entries_t in arms_of.items():
            succ[key] = list(entries_t)

        if not arms_of:
            return

        ipdom = _post_dominators(succ)
        for key, arms in arms_of.items():
            source, _, idx = key.partition(_FAN_SEP)
            join = ipdom[key]
            if join == _EXIT_NAME:
                raise TypeError(
                    f"{where}edges: the arms of the fan-out at {source!r} never meet — "
                    f"{', '.join(repr(a) for a in arms)} have no node they all reach before EXIT. "
                    f"Parallel arms converge on one node, and that node is where the flow carries on."
                )
            regions = tuple(self._region(arm, join) for arm in arms)
            self._check_region(where, source, arms, join, regions)
            fan = Fan(source=source, outcome=int(idx), arms=arms, join=join, regions=regions)
            self._check_no_other_way_in(where, fan, int(idx))
            self._fans[key] = fan
            self._fans_by_join.setdefault(join, []).append(fan)
            for entry, region in zip(arms, regions):
                scope = _Scope(nodes=region, source=source, entry=entry)
                for n in region:
                    current = self._scope.get(n)
                    # Regions nest, so the smallest one containing a node is the arm it
                    # actually belongs to.
                    if current is None or len(region) < len(current.nodes):
                        self._scope[n] = scope

    def _region(self, entry: str, join: str) -> frozenset[str]:
        """The nodes one arm owns: everything reachable from its entry without passing the
        join. The join itself is excluded — it belongs to whoever runs the fan-out."""
        seen: set[str] = set()
        frontier = [entry]
        while frontier:
            name = frontier.pop()
            if name == join or name in seen:
                continue
            seen.add(name)
            frontier.extend(self._targets_of(name))
        return frozenset(seen)

    def _check_region(
        self,
        where: str,
        source: str,
        arms: tuple[str, ...],
        join: str,
        regions: tuple[frozenset[str], ...],
    ) -> None:
        """A fan-out's arms have to be independent sub-graphs, because that is exactly what
        running them concurrently assumes: each arm runs its own nodes, once, and nothing
        else is in flight in it.

        The strong form of that, and the one the join's `depends()` rule leans on: control
        gets into an arm *only* by the fan-out taking it. Any other way in — a second route
        from the source, a jump from elsewhere, the graph's own entry sitting in there —
        would let the join be reached with only some of its arms behind it, which is exactly
        what `_arrivals` promises can't happen.
        """
        for i, region in enumerate(regions):
            if not region:
                # The arm's entry *is* the join, which means another arm reaches it: the two
                # are in sequence, not in parallel. Left alone the arm would own no nodes and
                # never stop, so it would run the join and everything after it a second time.
                raise TypeError(
                    f"{where}edges: the fan-out at {source!r} lists {arms[i]!r} as an arm, but that "
                    f"is also where its arms meet — the others reach it, so they don't run in "
                    f"parallel with it. Fan out to nodes that only meet further on."
                )
            if source in region:
                raise TypeError(
                    f"{where}edges: the arm {arms[i]!r} of the fan-out at {source!r} routes back to "
                    f"{source!r}. An arm ends at the join ({join!r}) — loop from there, not from "
                    f"inside the fan-out."
                )
            if self.entry in region:
                raise TypeError(
                    f"{where}edges: the graph begins at {self.entry!r}, which is inside the "
                    f"{arms[i]!r} arm of the fan-out at {source!r}. START would enter that arm "
                    f"without the fan-out, so the others would never run."
                )
            for j in range(i + 1, len(regions)):
                shared = sorted(region & regions[j])
                if shared:
                    raise TypeError(
                        f"{where}edges: the arms {arms[i]!r} and {arms[j]!r} of the fan-out at "
                        f"{source!r} both reach {', '.join(shared)}. Arms run at the same time, so "
                        f"they can't share a node before they meet at {join!r}."
                    )
            for node in sorted(region):
                for p in sorted(self._preds[node]):
                    if p in region:
                        continue
                    # The source is a legitimate predecessor of the arm's *entry* and of
                    # nothing else: routing into the middle of an arm skips its start.
                    if p == source and node == arms[i]:
                        continue
                    raise TypeError(
                        f"{where}edges: {p!r} routes to {node!r}, which is inside the "
                        f"{arms[i]!r} arm of the fan-out at {source!r}. An arm is only ever "
                        f"entered through its fan-out; route to {join!r} instead."
                    )

    def _check_no_other_way_in(self, where: str, fan: Fan, outcome: int) -> None:
        """The fan-out's own source must not reach its arms any other way.

        `edge(check).when(...).to(extra, common).otherwise(common)` reads as if `common`
        runs either way, and it does — but on the `.otherwise` route it runs *alone*, with
        no `extra` beside it. The join would then be reached with one arm missing, while
        every check above still passes: `common`'s only predecessor is the source either
        way, so nothing local can tell the two routes apart.
        """
        e = self._by_source[fan.source]
        elsewhere = {
            self._name_of(t)
            for i, other in enumerate(e.outcomes()) if i != outcome
            for t in other if not isinstance(t, _Exit)
        }
        inside = sorted(n for n in elsewhere & fan.nodes if n is not None)
        if inside:
            raise TypeError(
                f"{where}edges: {fan.source!r} also routes to {', '.join(inside)} outside its "
                f"fan-out to {', '.join(fan.arms)}, so {'that arm' if len(inside) == 1 else 'those arms'} "
                f"could run without the others. An arm runs only as part of its fan-out — give the "
                f"other branch its own node."
            )

    def _arrivals(
        self,
        node: str,
        scope: _Scope | None,
        available: Mapping[str, set[str]],
    ) -> set[str]:
        """What is guaranteed to have run when control arrives at `node` from within
        `scope` (one fan-out arm, or the whole flow when there is none).

        Two kinds of thing arrive at a node. A **fan-out** whose join this is delivers all
        of its arms at once, so it contributes the union of what each arm arrived with —
        recursing here per arm, which is what makes a nested fan-out ending at the same join
        come out right. An ordinary **predecessor** contributes what it alone guarantees.
        Control takes exactly one of those routes, so the node's own guarantee is their
        intersection.
        """
        parts: list[set[str]] = []
        consumed: set[str] = set()
        for fan in self._fans_by_join.get(node, ()):
            if not self._inside(fan.source, scope):
                continue
            consumed |= fan.nodes
            inner: set[str] = set()
            for entry, region in zip(fan.arms, fan.regions):
                inner |= self._arrivals(node, _Scope(region, fan.source, entry), available)
            parts.append(available[fan.source] | {fan.source} | inner)

        for p in self._preds[node]:
            if p in consumed:
                continue          # this predecessor arrived as part of a fan-out, above
            if scope is None:
                if self._scope.get(p) is not None:
                    continue
            elif p not in scope.nodes and not (p == scope.source and node == scope.entry):
                continue
            parts.append(available[p] | {p})
        return set.intersection(*parts) if parts else set()

    def _inside(self, name: str, scope: _Scope | None) -> bool:
        """Whether `name` sits strictly inside `scope` — at the flow's top level when there
        is no scope. A fan's own source is never inside its own arms, so this is also what
        stops `_arrivals` recursing into the fan it was called from."""
        return self._scope.get(name) is None if scope is None else name in scope.nodes

    def _check_cross_arm(
        self,
        where: str,
        required: Mapping[str, set[str]],
        optional: Iterable[tuple[str, str]],
    ) -> None:
        """A `depends()` between two arms of one fan-out is a race, not an ordering: the
        arms run at the same time, so whether the value is there — and which pass wrote it —
        is timing. The availability fixpoint rejects it anyway; this says why."""
        pairs = {(c, p) for c, deps in required.items() for p in deps} | set(optional)
        for consumer, producer in sorted(pairs):
            for fan in self._fans.values():
                here = next((i for i, r in enumerate(fan.regions) if consumer in r), None)
                there = next((i for i, r in enumerate(fan.regions) if producer in r), None)
                if here is None or there is None or here == there:
                    continue
                raise TypeError(
                    f"{where}edges: {consumer!r} depends on {producer!r}, but they sit in "
                    f"different arms of the fan-out at {fan.source!r}. Arms run at the same time, "
                    f"so neither can read the other — have {fan.join!r} read them both."
                )

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

        A fan-out's join flips the combining rule from intersection to *union*, and it has
        to: every arm runs, so every arm's work is behind the join. That is what makes the
        whole point of fanning out — one node reading what several arms produced — a legal
        `depends()` rather than a declaration error. Within an arm it's still intersection,
        since an arm can branch internally. `_arrivals` is that rule, and it recurses so a
        fan-out nested inside an arm that ends at the same join is handled the same way.
        """
        ordered = sorted(names)
        available: dict[str, set[str]] = {n: set(ordered) for n in ordered}
        available[self.entry] = set()

        changed = True
        while changed:
            changed = False
            for name in ordered:
                if name == self.entry:
                    continue          # START contributes nothing, so the entry stays empty
                fresh = self._arrivals(name, self._scope.get(name), available)
                if fresh != available[name]:
                    available[name] = fresh
                    changed = True

        self._check_cross_arm(where, required, optional)

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

    def inside_a_fan_out(self, name: str) -> bool:
        """Whether this node lives inside some fan-out's arms — i.e. whether it is only ever
        reached with sibling arms running beside it."""
        return name in self._scope

    def route(self, name: str, result: Any) -> Route:
        """Where control goes after `name` returned `result` — one node, `EXIT` (a `target`
        of None), or a `Fan` whose arms run concurrently before control resumes at its
        join. The runner asks this and nothing else about ordering."""
        e = self._by_source[name]
        index = e.resolve_index(result)
        fan = self._fans.get(f"{name}{_FAN_SEP}{index}")
        if fan is not None:
            return Route(target=None, fan=fan)
        return Route(target=self.target_name(e.outcomes()[index][0]), fan=None)

    def target_name(self, obj: Any) -> str | None:
        """Member name of an edge target, or None for `EXIT`."""
        return None if isinstance(obj, _Exit) else self._name_of(obj)


