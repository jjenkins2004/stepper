"""Control-flow edges: the other way a stage can declare its order.

A stage picks one of two models, and never mixes them:

- **No `edges`** — `depends()` is the order. `Scheduler` derives a DAG from it and runs
  independent steps concurrently. This is what every stage has always done.
- **`edges = (...)`** — you declare the whole graph, and it runs sequentially: one step,
  then whichever edge its result matches, until an edge reaches `EXIT`. Nothing is
  inferred. Every step in `steps` must appear in the graph, `START` says where it begins,
  and `depends()` becomes pure data wiring with no say in ordering.

The second model is what a loop needs, because `depends()` is dataflow and can't express
a cycle or a branch:

    steps = (edit, audit, analysis)

    edges = (
        edge(START).to(edit),
        edge(edit).to(audit),
        edge(audit).when(lambda r: r.passed).to(EXIT).otherwise(analysis),
        edge(analysis)
            .when(lambda r: r.rounds >= 8).to(EXIT)
            .when(lambda r: r.remeasures < 2).to(audit)
            .otherwise(edit),
    )

Steps stay exactly what they were: inputs via `depends()`, one output model. They know
nothing about routing — no marker types, no branch tokens in the return annotation, so
what lands on disk is still a real domain model. Declaring the edges *below* the steps is
what makes the back edge legal: `analysis -> edit` is a forward reference at that point in
the class body.

`edge(step)` is typed on that step's return model, so a predicate's only argument is the
value its source produced — `r.passed` above is checked against `AuditResult`, not `Any`.
The predicate is *called*, never inspected: nothing here reads a return type to decide a
route. Routing is only ever the edges you declared.

A branch sees its source's output and nothing else, on purpose. The step that just ran is
the thing reporting what happened, so whatever a decision needs belongs in its output
model — which also makes the persisted artifact a complete account of the state at that
point. A step needing a fact it doesn't own declares a `depends()` for it like any other
input.

**The framework counts nothing.** There is no round number, no loop state. Real loops
track several quantities at once — rounds, retries, budget raises — so a single
framework-owned counter would be one of them at best, and a second source of truth about
a loop that already records its own. Counters live in the step's output model, where
they're typed, persisted, and visible on disk; `r.rounds >= 8` above is a field the step
wrote. The one number the framework keeps is `Stage.max_steps`, a runaway fuse that is
never exposed to a predicate.

Predicates take the source's result and must be pure functions of it: they are
re-evaluated when a crashed run resumes, so no clock, no randomness, no network.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar, overload

if TYPE_CHECKING:
    from stepper.step import Step

R = TypeVar("R")


class _Start:
    """Sentinel edge source: where the graph begins. `edge(START).to(first_step)`."""

    __slots__ = ()
    name = "START"

    def __repr__(self) -> str:
        return "START"


class _Exit:
    """Sentinel edge target: the graph is finished and the stage is done."""

    __slots__ = ()
    name = "EXIT"

    def __repr__(self) -> str:
        return "EXIT"


START = _Start()
EXIT = _Exit()


class Edge(Generic[R]):
    """One source's outgoing control flow, built by `edge(source)`.

    `R` is the source step's output model, so every predicate's argument is typed:
    `edge(audit).when(lambda r: r.passed)` checks `passed` against `AuditResult`.

    Either unconditional (`.to(target)`) or a branch chain (`.when(pred).to(target)` one
    or more times, closed by `.otherwise(target)`). The chain is ordered — the first
    predicate that returns True wins — and `otherwise` is mandatory on a branch chain so
    exhaustiveness is structural rather than something we hope the lambdas cover.
    """

    __slots__ = ("source", "_branches", "_default", "_pending", "_unconditional")

    def __init__(self, source: "Step[R] | _Start") -> None:
        self.source = source
        self._branches: list[tuple[Callable[[R], bool], "Step[Any] | _Exit"]] = []
        self._default: "Step[Any] | _Exit | None" = None
        self._pending: Callable[[R], bool] | None = None
        self._unconditional = False

    def when(self, predicate: Callable[[R], bool]) -> "Edge[R]":
        """Take this branch when `predicate(result)` is True — `result` being whatever the
        source step just returned."""
        if self._unconditional:
            raise TypeError(f"edge({self.source.name}): .when(...) after an unconditional .to(...).")
        if self._pending is not None:
            raise TypeError(f"edge({self.source.name}): .when(...) twice with no .to(...) between them.")
        if self._default is not None:
            raise TypeError(f"edge({self.source.name}): .when(...) after .otherwise(...).")
        self._pending = predicate
        return self

    def to(self, target: "Step[Any] | _Exit") -> "Edge[R]":
        """Target of the pending `.when(...)`, or — with no pending predicate — this
        source's single unconditional target."""
        if self._pending is None:
            if self._branches or self._default is not None:
                raise TypeError(f"edge({self.source.name}): unconditional .to(...) mixed with branches; use .otherwise(...).")
            self._unconditional = True
            self._default = target
        else:
            self._branches.append((self._pending, target))
            self._pending = None
        return self

    def otherwise(self, target: "Step[Any] | _Exit") -> "Edge[R]":
        """Fallback when no `.when(...)` matched. Required on any branching edge."""
        if self._unconditional:
            raise TypeError(f"edge({self.source.name}): .otherwise(...) on an unconditional edge.")
        if not self._branches:
            raise TypeError(f"edge({self.source.name}): .otherwise(...) with no .when(...) before it.")
        if self._default is not None:
            raise TypeError(f"edge({self.source.name}): .otherwise(...) declared twice.")
        self._default = target
        return self

    @property
    def is_branching(self) -> bool:
        return bool(self._branches)

    def targets(self) -> list["Step[Any] | _Exit"]:
        return [t for _, t in self._branches] + ([self._default] if self._default is not None else [])

    def check(self) -> None:
        """Reject a half-built edge — raised while the stage class is being created."""
        if self._pending is not None:
            raise TypeError(f"edge({self.source.name}): .when(...) with no .to(...) after it.")
        if self._default is None:
            if not self._branches:
                raise TypeError(f"edge({self.source.name}): declared with no .to(...) — it routes nowhere.")
            raise TypeError(
                f"edge({self.source.name}): branching edge needs .otherwise(...) so every result routes somewhere."
            )

    def resolve(self, result: R) -> "Step[Any] | _Exit":
        """Where control goes after the source step returned `result`: the first branch
        whose predicate matches, else the default."""
        for predicate, target in self._branches:
            if predicate(result):
                return target
        assert self._default is not None      # check() ran at class creation
        return self._default


@overload
def edge(source: "Step[R]") -> Edge[R]: ...
@overload
def edge(source: _Start) -> Edge[Any]: ...
def edge(source: Any) -> Edge[Any]:
    """Start declaring a source's outgoing control flow — a `Step`, or `START` for the
    edge into the graph's first step. The returned `Edge` is typed on the step's output,
    so predicates get a typed `result`. See the module docstring."""
    return Edge(source)


class Graph:
    """A stage's validated control-flow graph: where it starts and the edge out of every
    step. Built once per stage class, so a malformed graph raises at class creation —
    same contract as the dependency DAG it replaces.

    "Declare the whole graph" is enforced here: every step the stage lists must have an
    edge out and be reachable from `START`, and `EXIT` must be reachable. Nothing is
    inferred and nothing is optional. On top of that, every step's required `depends()`
    must be *guaranteed* to have run before it — see `_check_deps_available`.
    """

    def __init__(self, edges: tuple[Edge[Any], ...], steps: tuple["Step[Any]", ...], label: str) -> None:
        self.label = label
        self._by_source: dict[str, Edge[Any]] = {}
        where = f"{label} " if label else ""
        # Membership is by *identity*, not name: two stages may each own a step called
        # "validate", and an edge naming the wrong stage's must be rejected, not quietly
        # rebound to the local one.
        member_steps = set(steps)
        members = {s.name for s in steps}
        entry: Any = None

        for e in edges:
            e.check()
            if isinstance(e.source, _Start):
                if entry is not None:
                    raise TypeError(f"{where}edges: more than one edge(START); the graph has one way in.")
                if e.is_branching:
                    raise TypeError(f"{where}edges: edge(START) must be unconditional — there is no result to branch on.")
                entry = e.targets()[0]
                if isinstance(entry, _Exit):
                    raise TypeError(f"{where}edges: edge(START).to(EXIT) runs nothing.")
                continue
            if e.source not in member_steps:
                raise TypeError(f"{where}edges: {e.source.name!r} is not in steps = (...).")
            if e.source.name in self._by_source:
                raise TypeError(f"{where}edges: {e.source.name!r} has more than one edge(...); one per step.")
            self._by_source[e.source.name] = e

        if entry is None:
            raise TypeError(f"{where}edges: no edge(START).to(...), so nothing says where the graph begins.")
        if entry not in member_steps:
            raise TypeError(f"{where}edges: edge(START) routes to {entry.name!r}, which is not in steps = (...).")
        self.entry: "Step[Any]" = entry

        for e in edges:
            for t in e.targets():
                if not isinstance(t, _Exit) and t not in member_steps:
                    raise TypeError(f"{where}edges: {e.source.name!r} routes to {t.name!r}, which is not in steps = (...).")

        missing = sorted(members - set(self._by_source))
        if missing:
            raise TypeError(
                f"{where}edges: {', '.join(missing)} declare no edge(...) out. A stage with edges "
                f"declares its whole graph — every step needs one."
            )

        self._check_reachability(where, members)
        self._check_deps_available(where, steps)

    def _predecessors(self, names: list[str]) -> dict[str, set[str]]:
        preds: dict[str, set[str]] = {n: set() for n in names}
        for source, e in self._by_source.items():
            for t in e.targets():
                if not isinstance(t, _Exit):
                    preds[t.name].add(source)
        return preds

    def _check_reachability(self, where: str, members: set[str]) -> None:
        """Every step must be reachable from the entry, and every step must be able to
        reach `EXIT` — otherwise the graph either contains dead code or contains a branch
        that, once taken, can never finish.

        The second half is per-step, not "somewhere in the graph there's an EXIT". A
        branch whose arm drops into a sub-cycle with no way out is exactly as broken as a
        graph with no EXIT at all; it just takes a run to find out.
        """
        seen: set[str] = set()
        frontier = [self.entry.name]
        while frontier:
            name = frontier.pop()
            if name in seen:
                continue
            seen.add(name)
            frontier.extend(t.name for t in self._by_source[name].targets() if not isinstance(t, _Exit))

        unreachable = sorted(members - seen)
        if unreachable:
            raise TypeError(f"{where}edges: {', '.join(unreachable)} is unreachable from START.")

        # Walk backwards from the steps that route to EXIT directly.
        can_exit = {
            n for n in members if any(isinstance(t, _Exit) for t in self._by_source[n].targets())
        }
        changed = True
        while changed:
            changed = False
            for name in members - can_exit:
                if any(t.name in can_exit for t in self._by_source[name].targets() if not isinstance(t, _Exit)):
                    can_exit.add(name)
                    changed = True

        stuck = sorted(members - can_exit)
        if stuck:
            raise TypeError(
                f"{where}edges: {', '.join(stuck)} has no route to EXIT — once control reaches "
                f"{'it' if len(stuck) == 1 else 'them'}, the graph could never finish."
            )

    def _reaches(self, source: str, target: str) -> bool:
        """Whether control can get from `source` to `target` by following edges — i.e.
        whether *some* execution runs `source` before `target`."""
        seen: set[str] = set()
        frontier = [t.name for t in self._by_source[source].targets() if not isinstance(t, _Exit)]
        while frontier:
            name = frontier.pop()
            if name == target:
                return True
            if name in seen:
                continue
            seen.add(name)
            frontier.extend(t.name for t in self._by_source[name].targets() if not isinstance(t, _Exit))
        return False

    def _check_deps_available(self, where: str, steps: tuple["Step[Any]", ...]) -> None:
        """A step's same-stage `depends()` has to line up with the graph, and the bar
        differs by kind:

        - **required** — the producer must have run on *every* path from START. A step
          fetches its inputs off the backend, so a producer that hasn't run has nothing
          stored and the fetch raises. `edge(START).to(b)` where `b = depends(a)` is
          impossible to run, not merely unusual.
        - **optional** — the producer must be able to run before the consumer on *at least
          one* path. `optional_depends` exists so a first pass can read None, not so a
          parameter can be None forever; if no route from producer to consumer exists, the
          wiring is dead and the graph doesn't mean what it looks like it means.

        The required side is a "must have executed" fixpoint: what's guaranteed at a step
        is the intersection over its predecessors of (guaranteed at that predecessor, plus
        the predecessor itself). The entry gets the empty set — START guarantees nothing —
        which is what makes a first-pass back-edge read come out unavailable, and why it
        has to be optional. The optional side is plain reachability, which is why a back
        edge passes: in `START -> b`, `b -> a`, `a -> b`, the second visit to `b` does have
        `a` behind it. Cross-stage deps are excluded from both: those are disk inputs from
        an earlier stage, not this graph's business.
        """
        names = [s.name for s in steps]
        member_steps = set(steps)
        required: dict[str, set[str]] = {}
        optional: list[tuple[str, str]] = []
        for s in steps:
            opt_params = s.optional_dependencies()
            deps = s.dependencies()
            # A producer that persists nothing is never fetched, so neither rule applies:
            # the dep is pure ordering, and in an edge-driven stage the edges already own
            # that. Nothing about the graph can make it fail.
            required[s.name] = {
                dep.name for param, dep in deps.items()
                if param not in opt_params and dep in member_steps and dep.model is not None
            }
            optional.extend(
                (s.name, dep.name) for param, dep in deps.items()
                if param in opt_params and dep in member_steps and dep.model is not None
            )

        preds = self._predecessors(names)
        available: dict[str, set[str]] = {n: set(names) for n in names}
        available[self.entry.name] = set()

        changed = True
        while changed:
            changed = False
            for name in names:
                if name == self.entry.name:
                    continue          # START contributes nothing, so the entry stays empty
                fresh = set.intersection(*(available[p] | {p} for p in preds[name]))
                if fresh != available[name]:
                    available[name] = fresh
                    changed = True

        for name in names:
            unmet = sorted(required[name] - available[name])
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

    def edge_for(self, name: str) -> Edge[Any]:
        return self._by_source[name]


def build_graph(edges: tuple[Edge[Any], ...], steps: tuple["Step[Any]", ...], label: str) -> Graph | None:
    """The stage's control-flow graph, or None when it declares no `edges` and therefore
    runs as a `depends()`-ordered DAG."""
    return Graph(edges, steps, label) if edges else None
