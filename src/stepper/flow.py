"""Flow: a step made of steps.

A flow has exactly a step's shape — declared inputs, one typed output — so a flow and a
step are interchangeable to whoever uses them. That symmetry is the whole design: because a
flow *has* an output, nobody ever reaches inside one, and because nobody reaches inside,
there is nothing to resolve, disambiguate, or guess.

    class TransformFlow(Flow[Clean]):
        raw = require(Raw)                                  # inputs

        @step
        async def strip(self, r=depends(raw)) -> Clean: ...

        @step
        async def check(self, c=depends(strip)) -> Clean: ...

        edges = (edge(START).to(strip), edge(strip).to(check), edge(check).to(EXIT))


    class PushFlow(Flow[Receipt]):
        @step
        async def fetch(self) -> Raw: ...

        transform = TransformFlow.bind(raw=fetch)           # bound where it's mounted

        @step
        async def send(self, c=depends(transform)) -> Receipt: ...
        #                       ^ depends on a FLOW, typed Clean, same as any step

        edges = (
            edge(START).to(fetch),
            edge(fetch).to(transform),
            edge(transform).to(send),
            edge(send).to(EXIT),
        )


    await PushFlow().run(run_id="run-1")

**Everything a step can read is declared on its own class.** A `depends()` names a step, a
nested flow, or one of this flow's `require()` inputs — nothing else is in scope, so wiring
never crosses a flow boundary. A value from further away arrives the only way it can: the
flow declares it needs one, and whoever mounts the flow says where it comes from.

**Closed and open.** A flow with no requirements is closed: it runs on its own, anywhere.
One with requirements is open — it runs wherever they're supplied. To run an open flow by
itself, write a small closed flow that supplies them; the harness is a flow like any other,
so its inputs persist too.

**Settings are injected, never constructed.** `SomeFlow()` takes nothing. The run's
`run_id`, backend and hooks are given to `run()` or `mount()` and reach every node from
there.

**Order is always declared.** Every flow spells out its graph, `START` to `EXIT`, and runs
it sequentially: run a node, ask its edge where to go, repeat. Targets may be steps *or*
nested flows, a predicate on either gets its typed output, and a cycle is how a flow loops.
Nothing is inferred from `depends()` — that is dataflow, and it is checked against the
graph rather than defining it.

**The output is whatever finishes.** The nodes with an edge to `EXIT` are the flow's
terminals; the one a run actually ends on supplies its value, persisted under the flow's own
path. `Flow[Clean]` is the contract, and every terminal has to satisfy it — checked when the
class is created, so a branch that finishes with the wrong model is a declaration error, not
a surprise mid-run.

**Addressing is a path.** `run("transform/strip")` is `transform` being asked to run
`strip`: each node eats the first segment and passes the rest down. `run()` runs everything.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from typing import Any, ClassVar, Generic, TypeVar, get_args, get_origin

from pydantic import BaseModel, ValidationError

from stepper.edges import Edge, Graph, Member
from stepper.hooks import Hooks
from stepper.node import Context, Node, join, split
from stepper.persist import DiskPersistService, PersistService
from stepper.producer import Producer, Requirement
from stepper.step import Step
from stepper.step_logging import format_module_end, format_module_start

_LOGGER = logging.getLogger(__name__)

# Where the loop cursor lands, under the flow's own path.
_CURSOR_KEY = "_loop_cursor"

R = TypeVar("R")


class _Cursor(BaseModel):
    """The loop's entire durable state: which node runs next.

    That really is all of it. Everything a resumed run needs was persisted by the steps
    themselves, so there is no snapshot to keep in sync and nothing here to drift out of
    date. `done` marks a graph that finished, so the next run starts clean instead of
    resuming a completed loop.
    """

    next: str | None
    done: bool = False


class FlowRef(Producer[R]):
    """A nested flow, declared with `SomeFlow.bind(**bindings)`.

    A producer like any other: its output is whatever the mounted flow ends on, so
    `depends(ref)` reads that value and is typed as it. The bindings say where the child's `require()`
    inputs come from — each one names a producer declared on *this* class, which is why a
    binding can never point somewhere the declaring flow can't see.
    """

    def __init__(self, target: "type[Flow[R]]", bindings: dict[str, Producer[Any]]) -> None:
        self.target = target
        self.bindings = bindings

    @property
    def model(self) -> Any:
        return self.target.output_model()

    def __repr__(self) -> str:
        return f"flow({self.target.__name__})"


def _declared_output(cls: "type[Flow[Any]]") -> Any:
    """The model in `class X(Flow[Model])`, or None for a bare `Flow`. Read off this
    class's own bases — an inherited `__orig_bases__` belongs to something else."""
    for base in vars(cls).get("__orig_bases__", ()):
        if get_origin(base) is Flow:
            args = get_args(base)
            if args and isinstance(args[0], type):
                return args[0]
    return None


def _normalize_hooks(hooks: Hooks | Sequence[Hooks] | None) -> tuple[Hooks, ...]:
    """Coerce the public `hooks` arg to an internal tuple: None -> (), one -> a 1-tuple."""
    if hooks is None:
        return ()
    if isinstance(hooks, Sequence):
        return tuple(hooks)
    return (hooks,)


class _Plan:
    """One flow class's declaration, worked out once when the class is created: its nodes,
    what waits on what, and the ordering model built from that.

    All of it is local. A flow's nodes are its own attributes and its edges are its own
    bindings, so nothing here has to look outside the class — which is why this can run at
    class creation rather than at mount.
    """

    def __init__(self, cls: "type[Flow[Any]]") -> None:
        self.owner = cls
        self.label = cls.__name__
        self.max_steps = cls.max_steps
        producers = cls.producers()
        self.requirements = {n: p for n, p in producers.items() if isinstance(p, Requirement)}
        self.members: list[Member] = [
            (n, p) for n, p in producers.items() if not isinstance(p, Requirement)
        ]
        member_set = {id(p) for _, p in self.members}

        self._check_wiring(producers)

        required: dict[str, set[str]] = {}
        optional: list[tuple[str, str]] = []
        for name, node in self.members:
            req: set[str] = set()
            if isinstance(node, Step):
                opt_params = node.optional_dependencies()
                for param, dep in node.dependencies().items():
                    if id(dep) not in member_set or dep.name == name:
                        continue          # a requirement, or itself — neither is an edge here
                    # A producer that persists nothing is never fetched, so the required /
                    # optional rules don't apply to it: the dep is pure ordering, which the
                    # graph already owns.
                    if dep.model is None:
                        continue
                    (optional.append((name, dep.name)) if param in opt_params else req.add(dep.name))
            else:
                # Whatever a child flow's bindings read has to have run before it.
                for bound in node.bindings.values():
                    if id(bound) in member_set and bound.name != name and bound.model is not None:
                        req.add(bound.name)
            required[name] = req

        if not cls.edges:
            raise TypeError(
                f"{self.label}: a flow declares its order — `edges = (edge(START).to(...), ..., "
                f"edge(...).to(EXIT))`. Nothing is inferred from depends()."
            )
        self.graph = Graph(cls.edges, self.members, self.label, required, optional)
        if self.max_steps < 1:
            raise TypeError(f"{self.label}: max_steps must be at least 1.")
        self.output_model = self._check_output()

    def _check_output(self) -> Any:
        """A flow's value is whichever terminal the run ended on, so every terminal has to
        produce the same thing — and, when the class says `Flow[Model]`, that thing."""
        by_name = dict(self.members)
        models = {name: by_name[name].model for name in self.graph.terminals()}
        distinct = {m for m in models.values()}
        if len(distinct) > 1:
            spread = ", ".join(
                f"{n} -> {getattr(m, '__name__', m)}" for n, m in sorted(models.items())
            )
            raise TypeError(
                f"{self.label}: the nodes that reach EXIT don't agree on what this flow "
                f"produces ({spread}). Every terminal has to end with the same model."
            )
        actual = distinct.pop() if distinct else None
        declared = _declared_output(self.owner)
        if declared is not None and actual is not declared:
            raise TypeError(
                f"{self.label}: declared Flow[{declared.__name__}] but its terminals produce "
                f"{getattr(actual, '__name__', actual)}."
            )
        return actual

    def _check_wiring(self, producers: dict[str, Producer[Any]]) -> None:
        """Everything a flow names has to be declared on the flow. That single rule is what
        confines wiring to one class, so it's worth a precise error rather than a stray
        KeyError somewhere in a run."""
        if not self.members:
            raise TypeError(f"{self.label}: a flow needs nodes — @step methods, nested flows, or both.")
        mine = {id(p) for p in producers.values()}
        # Two names for one producer would give two nodes that resolve to the same key,
        # silently. Identity is how edges and depends() find a node, so it has to be 1:1.
        seen: dict[int, str] = {}
        for name, p in producers.items():
            if id(p) in seen:
                raise TypeError(
                    f"{self.label}: {name!r} and {seen[id(p)]!r} are the same {p!r}. Declare it "
                    f"once, or mount two separate ones."
                )
            seen[id(p)] = name

        for name, node in self.members:
            if isinstance(node, Step):
                for param, dep in node.dependencies().items():
                    if id(dep) not in mine:
                        raise TypeError(
                            f"{self.label}.{name}: {param!r} depends on {dep!r}, which belongs to "
                            f"another flow. A step reads its own flow's steps, nested flows, and "
                            f"require() inputs — nothing else. Add a require() and let whoever "
                            f"mounts {self.label} bind it."
                        )
                continue
            wanted = node.target.plan().requirements
            for field, bound in node.bindings.items():
                if field not in wanted:
                    raise TypeError(
                        f"{self.label}.{name}: {node.target.__name__} has no require() named "
                        f"{field!r}. It declares: {', '.join(wanted) or '(none)'}."
                    )
                if id(bound) not in mine:
                    raise TypeError(
                        f"{self.label}.{name}: {field!r} is bound to {bound!r}, which is not "
                        f"declared on {self.label}."
                    )
                want, got = wanted[field].model, bound.model
                if want is not None and got is not want:
                    raise TypeError(
                        f"{self.label}.{name}: {field!r} wants {getattr(want, '__name__', want)} "
                        f"but {bound!r} produces {getattr(got, '__name__', got)}."
                    )
            unbound = sorted(set(wanted) - set(node.bindings))
            if unbound:
                raise TypeError(
                    f"{self.label}.{name}: {node.target.__name__} requires "
                    f"{', '.join(unbound)}, which {self.label} does not bind."
                )

        # The cursor shares the flow's key namespace, so a node of that name would
        # overwrite it (and be overwritten by it) on every pass.
        if any(name == _CURSOR_KEY for name, _ in self.members):
            raise TypeError(f"{self.label}: {_CURSOR_KEY!r} is reserved for the loop cursor; rename the node.")


class Flow(Node, Generic[R]):
    """A step made of steps: declared inputs, one typed output, and nodes in between.

    Subclass it, `require()` what it needs, write `@step` methods and `flow(...)` children,
    and name one of them `output`. `Flow[Clean]` says what that output is.
    """

    edges: ClassVar[tuple[Edge[Any], ...]] = ()      # control flow; declare it and it owns ordering
    max_steps: ClassVar[int] = 1000                  # runaway fuse for an edge-driven flow
    flow_name: ClassVar[str] = "flow"                # node name, from the class name
    _producers: ClassVar[dict[str, Producer[Any]]] = {}
    _class_plan: ClassVar["_Plan | None"] = None

    def __init_subclass__(cls) -> None:
        """Read the class body — its `require()`s, `@step`s and `flow()`s — and validate the
        whole declaration. Everything a flow says is about itself, so all of it can be
        checked here, before any instance exists."""
        concrete = next((b for b in cls.__mro__[1:] if getattr(b, "_class_plan", None)), None)
        if concrete is not None:
            raise TypeError(
                f"{cls.__name__} subclasses {concrete.__name__}, which is already a flow. A flow "
                f"is composed, not inherited — mount it with {concrete.__name__}.bind(...) instead. "
                f"(Its steps belong to it, so a subclass would have none of its own.)"
            )
        if "__init__" in vars(cls):
            raise TypeError(
                f"{cls.__name__} defines __init__. A flow's inputs are declared with require() "
                f"and its settings are injected when it runs, so there is nothing to construct."
            )
        cls.flow_name = cls.__name__.removesuffix("Flow").lower() or cls.__name__.lower()
        cls._producers = {
            name: value for name, value in vars(cls).items() if isinstance(value, Producer)
        }
        for p in cls._producers.values():
            if isinstance(p, Step):
                p.claim(cls)
        cls._class_plan = _Plan(cls)

    @classmethod
    def producers(cls) -> dict[str, Producer[Any]]:
        """Everything this class declares that has an output, by attribute name — the whole
        of what its steps are allowed to name."""
        return cls._producers

    @classmethod
    def bind(cls, **bindings: Any) -> "FlowRef[R]":
        """Mount this flow inside another, wiring its `require()` inputs to producers
        declared there — the call, to `require()`'s parameters:

            upload = UploadFlow.bind(listing=listing)

        Each keyword is one of this flow's requirements; each value is a producer declared
        on the mounting class. Both are checked when that class is created.
        """
        return FlowRef(cls, dict(bindings))

    @classmethod
    def output_model(cls) -> Any:
        """What this flow produces: the model every one of its terminals ends with."""
        return cls.plan().output_model

    @classmethod
    def plan(cls) -> _Plan:
        if cls._class_plan is None:
            raise TypeError(f"{cls.__name__} declares no nodes.")
        return cls._class_plan

    def __init__(self) -> None:
        """Takes nothing. A flow's inputs are `require()` declarations bound by whoever
        mounts it, and the run's settings are given to `run()` or `mount()`."""
        super().__init__()
        self.name = type(self).flow_name
        self._bound_nodes: dict[str, Node] = {}
        # Set when a parent mounts this flow: which flow bound it, and with what.
        self._binder: tuple[Flow[Any], FlowRef[Any]] | None = None

    # --- mounting ---------------------------------------------------------------------

    def mount(
        self,
        *,
        run_id: str | None = None,
        persist_service: PersistService | None = None,
        hooks: Hooks | Sequence[Hooks] | None = None,
    ) -> "Flow[R]":
        """Make this flow the root of a run and hand every node under it a path and the
        run's settings.

        Args:
            run_id: Labels this run. It prefixes persist keys, so repeat runs never
                overwrite each other, and leaves node paths alone so an address means the
                same thing in every run.
            persist_service: Backend for the run — it owns where output physically lands.
                Defaults to `DiskPersistService(base_dir="output")`.
            hooks: One `Hooks` or a sequence of them, wrapping every node in the tree.
        """
        if self.is_bound:
            return self
        cls = type(self)
        needs = cls.plan().requirements
        if needs:
            raise TypeError(
                f"{cls.__name__} requires {', '.join(needs)}, so it can't be the root of a run. "
                f"Mount it inside a flow that binds them, or write a small flow that supplies "
                f"them and run that."
            )
        self._attach(
            path=cls.flow_name,
            ctx=Context(
                persist=persist_service or DiskPersistService(base_dir=Path("output")),
                hooks=_normalize_hooks(hooks),
                run_id=run_id,
            ),
            parent=None,
            binder=None,
        )
        return self

    def _attach(
        self,
        *,
        path: str,
        ctx: Context,
        parent: "Flow[Any] | None",
        binder: "tuple[Flow[Any], FlowRef[Any]] | None",
    ) -> None:
        """Give this flow its place in the run and build everything under it. Children are
        constructed here, not earlier: a flow declaration is inert until something mounts
        it, so the same class can appear in two runs without sharing anything."""
        self._path = path
        self._ctx = ctx
        self._parent = parent
        self._binder = binder
        for name, node in type(self).plan().members:
            if isinstance(node, Step):
                self._bound_nodes[name] = node._bind(path=join(path, name), ctx=ctx, parent=self)
            else:
                child = node.target()
                child._attach(path=join(path, name), ctx=ctx, parent=self, binder=(self, node))
                self._bound_nodes[name] = child

    # --- resolution -------------------------------------------------------------------

    def key_of(self, producer: Producer[Any]) -> str:
        """Where a producer's value lives, as a node path.

        Three cases, and every one of them is a lookup rather than a search: a step is at
        its own path; a nested flow is wherever *its* output is; a requirement is whatever
        the flow above bound it to, asked in that flow's own terms. Bindings only ever point
        outward, so this terminates at a step.
        """
        if isinstance(producer, Step):
            return join(self._path, producer.name)
        if isinstance(producer, FlowRef):
            # A flow persists what it ended on under its own path, so it reads like a step.
            return self._bound_nodes[producer.name].path
        if isinstance(producer, Requirement):
            if self._binder is None:
                raise TypeError(f"{self.path}: {producer.name!r} was never bound.")
            parent, ref = self._binder
            return parent.key_of(ref.bindings[producer.name])
        raise TypeError(f"{self.path}: {producer!r} is not a producer this flow can read.")

    # --- inspection -------------------------------------------------------------------

    def node(self, target: str) -> Node:
        """The bound node at `target`, resolved the same way `run()` resolves it."""
        head, rest = split(target)
        node = self._bound_nodes.get(head)
        if node is None:
            raise ValueError(f"{self.path}: no node {head!r} (in {target!r}).")
        if not rest:
            return node
        if isinstance(node, Flow):
            return node.node(rest)
        raise ValueError(f"{node.path}: {rest!r} addresses a node inside a step, and a step is a leaf.")

    def describe(self) -> dict[str, Any]:
        plan = type(self).plan()
        return {
            "path": self._path,
            "kind": "flow",
            "flow": type(self).__name__,
            "requires": {n: p.model.__name__ for n, p in plan.requirements.items()},
            "output": getattr(plan.output_model, "__name__", None),
            "terminals": plan.graph.terminals(),
            "nodes": [n.describe() for n in self._bound_nodes.values()],
        }

    # --- edge-driven ordering ---------------------------------------------------------

    def _cursor_key(self) -> str:
        return self.ctx.key(join(self._path, _CURSOR_KEY))

    def _load_cursor(self) -> _Cursor | None:
        """The saved cursor, or None when there's nothing to resume — no cursor yet, a
        graph that already finished, or a checkpoint too damaged to trust. A damaged cursor
        restarts from `START` rather than failing the run: the nodes' own output is the real
        state, and this is only a pointer into it.

        Only *those* failures are absorbed. A backend that can't be read at all is a real
        problem, and swallowing it would silently re-run every node from the top —
        duplicating exactly the side effects the resume contract asks callers to guard."""
        try:
            cursor = self.ctx.persist.fetch(self._cursor_key(), _Cursor)
        except FileNotFoundError:
            return None                 # nothing checkpointed yet — a normal first run
        except ValidationError:
            _LOGGER.warning("%s: loop cursor is unreadable; starting from START.", self.path)
            return None
        if cursor.done or cursor.next is None:
            return None
        if cursor.next not in self._bound_nodes:
            return None      # graph was edited since the checkpoint — start clean
        return cursor

    def _save_cursor(self, cursor: _Cursor) -> None:
        """Checkpoint progress. Best-effort: a failed write must never fail a node, since
        the cursor is an optimization and the persisted output is the real state."""
        try:
            self.ctx.persist.persist(self._cursor_key(), cursor, _Cursor)
        except Exception:
            _LOGGER.warning("%s: could not checkpoint the graph cursor.", self.path)

    async def _run_graph(self, start_at: str | None = None) -> Any:
        """Run the declared graph to completion and return this flow's output value (or,
        with no `output`, the last value of each node this call ran, in declaration order).

        Sequential by construction: run a node, ask its edge where to go, checkpoint, go.
        The only number tracked is `executed`, and it is never exposed — it's the fuse that
        stops a wrong predicate from spinning forever, not a round counter. Anything the
        graph routes on is a field the steps themselves wrote.

        `start_at` enters the graph at a chosen node, overriding both the saved cursor and
        the entry — the manual counterpart to a crash resume. It runs against whatever is
        persisted, so a node whose required inputs were never written raises as it would
        anywhere else.
        """
        plan = type(self).plan()
        graph = plan.graph
        cursor = None if start_at is not None else self._load_cursor()
        if start_at is not None:
            node = start_at
        elif cursor is not None and cursor.next is not None:
            node = cursor.next
        else:
            node = graph.entry
        results: dict[str, Any] = {}
        executed = 0

        if start_at is not None:
            _LOGGER.info("%s: entering the graph at %s.", self.path, node)
            # Checkpoint the entry *before* running it. Otherwise a crash in that first node
            # would leave whatever cursor was already there, and the next plain run would
            # resume somewhere this run never intended to be.
            self._save_cursor(_Cursor(next=node))
        elif cursor is not None:
            _LOGGER.info("%s: resuming at %s.", self.path, node)

        while True:
            executed += 1
            if executed > plan.max_steps:
                raise RuntimeError(
                    f"{self.path}: ran {plan.max_steps} nodes without reaching EXIT "
                    f"(max_steps is a runaway fuse — a predicate is probably wrong)."
                )
            results[node] = await self._bound_nodes[node].run()

            target = graph.target_name(graph.edge_for(node).resolve(results[node]))
            if target is None:                       # EXIT
                self._save_cursor(_Cursor(next=None, done=True))
                return self._finish(results[node])

            node = target
            self._save_cursor(_Cursor(next=node))

    # --- running ----------------------------------------------------------------------

    def _finish(self, value: Any) -> Any:
        """A flow persists what it ended on under its own path, so reading a flow is reading
        a key like any other and a finished subtree leaves one artifact naming it."""
        model = type(self).plan().output_model
        if model is not None:
            self.ctx.persist.persist(self.ctx.key(self._path), value, model)
        return value

    async def run(
        self,
        target: str | None = None,
        *,
        follow_edges: bool = False,
        run_id: str | None = None,
        persist_service: PersistService | None = None,
        hooks: Hooks | Sequence[Hooks] | None = None,
    ) -> Any:
        """Run this flow, or the node at `target` beneath it, and return its output.

        Args:
            target: A `/`-separated node path (`"transform"`, `"transform/strip"`). Omit it
                to run every node. This flow resolves the first segment and hands the rest
                down, so the node that owns the last segment is the one that acts. A path
                landing on a flow runs that whole subtree; one landing on a step runs that
                step alone, against whatever is currently persisted.
            follow_edges: Instead of running `target` alone, *enter the graph* of the flow
                that owns it and keep going until an edge reaches `EXIT` — the manual
                counterpart to a crash resume. Needs a flow that declares `edges`.
            run_id, persist_service, hooks: The run's settings, used only when this call is
                what mounts the flow.
        """
        if self.is_bound and (run_id is not None or persist_service is not None or hooks is not None):
            raise TypeError(
                f"{self.path}: the run's settings belong to whatever mounted it. Pass them to "
                f"mount(), or to run() on an unmounted root."
            )
        self.mount(run_id=run_id, persist_service=persist_service, hooks=hooks)

        if target:
            head, rest = split(target)
            node = self._bound_nodes.get(head)
            if node is None:
                raise ValueError(f"{self.path}: no node {head!r} (in {target!r}).")
            if rest:
                return await node.run(rest, follow_edges=follow_edges)
            if follow_edges:
                return await self._run_graph(start_at=head)
            return await node.run()

        if follow_edges:
            raise ValueError("follow_edges needs a target to start from.")
        return await self._run_all()

    async def _run_all(self) -> Any:
        """Follow this flow's graph from `START` (or its saved cursor) until an edge reaches
        `EXIT`, and return what it ended on."""
        ctx = self.ctx
        _LOGGER.info(format_module_start(module_name=self.path, step_count=len(self._bound_nodes)))
        started = perf_counter()
        try:
            with ExitStack() as stack:
                # Fan out to every hook's flow() (enter in order, exit in reverse).
                for hook in ctx.hooks:
                    stack.enter_context(
                        hook.flow(path=self.path, node_count=len(self._bound_nodes))
                    )
                return await self._run_graph()
        finally:
            elapsed_ms = int((perf_counter() - started) * 1000)
            _LOGGER.info(format_module_end(module_name=self.path, elapsed_ms=elapsed_ms))


__all__ = ["Flow", "FlowRef"]
