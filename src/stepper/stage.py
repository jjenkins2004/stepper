"""Stage: a set of steps sharing one PersistService, run by a dependency scheduler.

Subclass `Stage`, mark async methods with `@step`, and list them in `steps = (...)`.
The tuple is just the *membership* set — which steps belong to the stage. Run order
comes from each step's `depends()` markers: `Scheduler` turns the tuple into a
validated DAG, and `run_steps()` follows it — launching every step whose upstreams
have finished, concurrently, so independent steps run at the same time.

Each step persists its return value under `<stage_name>/<step>`; wire an input by
defaulting a parameter to `depends(producer_step)`, which fetches that step's
persisted value (typed as its return type). A `depends()` may point at a step on
another stage — that value is already on disk from the earlier stage, so it's an
input, not a scheduling edge within this stage.

A stage that declares `edges = (...)` below its steps uses the other ordering model
(see `edges.py`): it runs its steps **sequentially**, routed by the graph it declares,
from `edge(START).to(...)` until an edge reaches `EXIT` — which is how a stage expresses a
loop, since `depends()` is dataflow and can't express a cycle or a branch. The two models
never mix: with `edges` the whole graph is declared and `depends()` is pure data wiring
with no say in ordering; without them, `depends()` is the order and the DAG runs
concurrently as it always has.

An edge-driven step persists under the same `<stage_name>/<step>` key on every pass — it
overwrites, so a `depends()` always reads the latest value. That's the correct read in a
sequential graph: an upstream in the same pass has already written it, and a step reading
across passes (via `optional_depends`) wants the previous pass's value, which is exactly
what's there.

Progress is checkpointed to `<stage_name>/_loop_cursor` after every step — the name of the
next step, and nothing else. There is no loop state to save: every value a resumed run
needs was already persisted by the steps themselves, and any counter the graph routes on
is a field the steps wrote. A crashed run re-entering the stage resumes at that step
rather than at `START`. Resume restores *values*, not side effects: the step it restarts
at may have already half-run, so steps in an edge-driven stage must be re-entrant.

Add tracing/telemetry by passing a `Hooks` implementation, or several as a sequence
(default: no-op); each step and each stage run is wrapped in every hook's matching
context manager (entered in order, exited in reverse).
"""

import logging
from collections.abc import Sequence
from contextlib import ExitStack
from time import perf_counter
from typing import Any, Callable, ClassVar, Coroutine

from pydantic import BaseModel, ValidationError

from stepper.edges import Edge, Graph, _Exit, build_graph
from stepper.hooks import Hooks
from stepper.persist import PersistService
from stepper.scheduler import Scheduler
from stepper.step import Step
from stepper.step_logging import (
    format_module_end,
    format_module_start,
    format_step_end,
    format_step_fail,
    format_step_start,
)

_LOGGER = logging.getLogger(__name__)

# Where the loop cursor lands, under the stage's own namespace.
_CURSOR_KEY = "_loop_cursor"


class _Cursor(BaseModel):
    """The loop's entire durable state: which step runs next.

    That really is all of it. Everything a resumed run needs was persisted by the steps
    themselves, so there is no snapshot to keep in sync and nothing here to drift out of
    date. `done` marks a graph that finished, so the next run starts clean instead of
    resuming a completed loop.
    """

    next: str | None
    done: bool = False


def _normalize_hooks(hooks: Hooks | Sequence[Hooks] | None) -> tuple[Hooks, ...]:
    """Coerce the public `hooks` arg to an internal tuple: None -> (), a lone `Hooks` -> a
    1-tuple, a sequence -> its tuple. A single hook then runs exactly as it did before."""
    if hooks is None:
        return ()
    if isinstance(hooks, Sequence):
        return tuple(hooks)
    return (hooks,)


class Stage:
    """Base stage: declare `steps = (...)` (membership). Run order comes from `depends()`
    — a concurrent DAG — unless the stage declares `edges = (...)`, in which case it runs
    sequentially through the graph those edges spell out."""

    stage_name: ClassVar[str] = ""             # inferred from the class name
    steps: ClassVar[tuple[Step[Any], ...]] = ()  # membership set — NOT run order (that's from deps)
    edges: ClassVar[tuple[Edge[Any], ...]] = ()  # control flow; declare it and it owns ordering
    max_steps: ClassVar[int] = 1000            # runaway fuse for an edge-driven stage
    _graph: ClassVar[Graph | None]             # validated control flow, or None for a DAG stage
    _scheduler: ClassVar[Scheduler | None]     # dependency runner; None for an edge-driven stage

    def __init_subclass__(cls) -> None:
        cls._check_steps()
        cls.stage_name = cls.__name__.removesuffix("Stage")   # "ExtractStage" -> "Extract"

        for step in cls.steps:
            step.claim(cls)

        # After claim(), so cross-stage deps resolve. Exactly one ordering model is built,
        # and either way it validates now, before any run: the graph (unknown target,
        # missing `otherwise`, step with no edge out, unreachable step, no route to EXIT)
        # or the dep DAG (unknown target, cycle).
        cls._graph = build_graph(cls.edges, cls.steps, cls.__name__)
        if cls._graph is not None and cls.max_steps < 1:
            raise TypeError(f"{cls.__name__}: max_steps must be at least 1.")
        cls._scheduler = None if cls._graph is not None else Scheduler(cls.steps, label=cls.__name__)

    @classmethod
    def _check_steps(cls) -> None:
        """Reject a stage missing `steps`, listing a non-Step, or repeating a step
        (which would run and overwrite its own output)."""
        if not cls.steps or not all(isinstance(s, Step) for s in cls.steps):
            raise TypeError(f"{cls.__name__} must declare `steps = (...)` listing its steps.")
        if len(set(cls.steps)) != len(cls.steps):
            raise TypeError(f"{cls.__name__}.steps has a duplicate step.")
        # The cursor shares the stage's key namespace, so a step of that name would
        # overwrite it (and be overwritten by it) on every pass.
        if any(s.name == _CURSOR_KEY for s in cls.steps):
            raise TypeError(f"{cls.__name__}: {_CURSOR_KEY!r} is reserved for the loop cursor; rename the step.")

    def __init__(
        self, *, persist_service: PersistService, hooks: Hooks | Sequence[Hooks] | None = None
    ) -> None:
        """
        Args:
            persist_service: Backend each step fetches its inputs from and persists its
                output to.
            hooks: One `Hooks`, a sequence of them, or None (default: no-op). Several are
                fanned out — entered in order, exited in reverse — and each receives the
                step output directly. A Pipeline overrides this only when the Pipeline
                itself was given hooks.
        """
        self._persist = persist_service
        self._hooks: tuple[Hooks, ...] = _normalize_hooks(hooks)
        # name -> runner, so run_step/get_steps can target one step by name
        self._runners: dict[str, Callable[[], Coroutine[Any, Any, Any]]] = {
            step.name: self._get_runner_for(step) for step in self.steps
        }

    @staticmethod
    def _key_for(s: Step[Any]) -> str:
        return "/".join((s.get_owner().stage_name, s.name))

    def _get_runner_for(self, step: Step[Any]) -> Callable[[], Coroutine[Any, Any, Any]]:
        """Return a coroutine that runs the step: fetch inputs, run, persist, and log.
        Identical whether the stage orders itself by edges or by deps — a step on a cycle
        is just this same runner called again."""

        async def run() -> Any:
            deps = step.dependencies()
            optional = step.optional_dependencies()
            input_type = ", ".join(dep.model.__name__ if dep.model is not None else "None" for dep in deps.values()) or "None"
            output_type = step.model.__name__ if step.model is not None else "None"
            _LOGGER.info(format_step_start(step_name=step.name, input_type=input_type, output_type=output_type))
            started = perf_counter()
            try:
                with ExitStack() as stack:
                    # Enter every hook's step() in order (ExitStack exits them in reverse).
                    # One hook behaves exactly as before; () is a clean no-op. A raise in
                    # __enter__/__exit__ unwinds the already-entered hooks with it.
                    reports = [
                        stack.enter_context(
                            hook.step(
                                stage_name=self.stage_name,
                                step_name=step.name,
                                input_type=input_type,
                                output_type=output_type,
                            )
                        )
                        for hook in self._hooks
                    ]
                    # Grab each declared input. A required dep with no persisted value
                    # raises (as always); an optional dep with none reads back as None.
                    inputs: dict[str, Any] = {}
                    for name, dep in deps.items():
                        if dep.model is None:
                            # Producer persists nothing, so there is nothing to fetch —
                            # the dep is pure ordering and the parameter is None. Typed
                            # that way too, since `depends()` on a `Step[None]` is None.
                            inputs[name] = None
                            continue
                        try:
                            inputs[name] = self._persist.fetch(self._key_for(dep), dep.model)
                        except FileNotFoundError:
                            if name not in optional:
                                raise
                            inputs[name] = None

                    # Run the step once
                    result = await step.fn(self, **inputs)

                    # Persist the result if the step declares an output model, then hand it
                    # to each hook's StepReport (skipping any hook that yielded None).
                    if step.model is not None:
                        self._persist.persist(self._key_for(step), result, step.model)
                        for report in reports:
                            if report is not None:
                                report.set_output(result)
            except Exception as exc:
                elapsed_ms = int((perf_counter() - started) * 1000)
                _LOGGER.exception(format_step_fail(step_name=step.name, elapsed_ms=elapsed_ms, error_type=type(exc).__name__))
                raise

            elapsed_ms = int((perf_counter() - started) * 1000)
            _LOGGER.info(format_step_end(step_name=step.name, elapsed_ms=elapsed_ms, output_type=type(result).__name__))
            return result

        return run

    # --- edge-driven ordering ---------------------------------------------------------

    def _load_cursor(self) -> _Cursor | None:
        """The saved cursor, or None when there's nothing to resume — no cursor yet, a
        graph that already finished, or a checkpoint too damaged to trust. A damaged
        cursor restarts from `START` rather than failing the run: the steps' own output is
        the real state, and this is only a pointer into it.

        Only *those* failures are absorbed. A backend that can't be read at all is a real
        problem, and swallowing it would silently re-run every step from the top —
        duplicating exactly the side effects the resume contract asks callers to guard."""
        try:
            cursor = self._persist.fetch(f"{self.stage_name}/{_CURSOR_KEY}", _Cursor)
        except FileNotFoundError:
            return None                 # nothing checkpointed yet — a normal first run
        except ValidationError:
            _LOGGER.warning("%s: loop cursor is unreadable; starting from START.", self.stage_name)
            return None
        if cursor.done or cursor.next is None:
            return None
        if self._graph is None or cursor.next not in self._runners:
            return None      # graph was edited since the checkpoint — start clean
        return cursor

    def _save_cursor(self, cursor: _Cursor) -> None:
        """Checkpoint progress. Best-effort: a failed write must never fail a step, since
        the cursor is an optimization and the steps' persisted output is the real state."""
        try:
            self._persist.persist(f"{self.stage_name}/{_CURSOR_KEY}", cursor, _Cursor)
        except Exception:
            _LOGGER.warning("%s: could not checkpoint the graph cursor.", self.stage_name)

    async def _run_graph(self, start_at: str | None = None) -> list[Any]:
        """Run the declared graph to completion and return the last value of each step
        *this call ran*, in declaration order — the same shape, and the same partial-run
        caveat, as a DAG run whose steps didn't all execute. A resumed run therefore
        returns only the tail it executed; read the rest back off the PersistService.

        Sequential by construction: run a step, ask its edge where to go, checkpoint, go.
        The only number tracked is `executed`, and it is never exposed — it's the fuse
        that stops a wrong predicate from spinning forever, not a round counter. Anything
        the graph routes on is a field the steps themselves wrote.

        `start_at` enters the graph at a chosen step, overriding both the saved cursor and
        the entry — the manual counterpart to a crash resume. It runs against whatever is
        persisted, so a step whose required inputs were never written raises as it would
        anywhere else.
        """
        graph = self._graph
        assert graph is not None                  # only called for an edge-driven stage
        cursor = None if start_at is not None else self._load_cursor()
        if start_at is not None:
            node = start_at
        elif cursor is not None and cursor.next is not None:
            node = cursor.next
        else:
            node = graph.entry.name
        results: dict[str, Any] = {}
        executed = 0

        if start_at is not None:
            _LOGGER.info("%s: entering the graph at %s.", self.stage_name, node)
            # Checkpoint the entry *before* running it. Otherwise a crash in that first
            # step would leave whatever cursor was already there, and the next plain run
            # would resume somewhere this run never intended to be.
            self._save_cursor(_Cursor(next=node))
        elif cursor is not None:
            _LOGGER.info("%s: resuming at %s.", self.stage_name, node)

        while True:
            executed += 1
            if executed > self.max_steps:
                raise RuntimeError(
                    f"{self.stage_name}: ran {self.max_steps} steps without reaching EXIT "
                    f"(max_steps is a runaway fuse — a predicate is probably wrong)."
                )
            results[node] = await self._runners[node]()

            target = graph.edge_for(node).resolve(results[node])
            if isinstance(target, _Exit):     # same predicate the graph validated with
                self._save_cursor(_Cursor(next=None, done=True))
                return [results[s.name] for s in self.steps if s.name in results]

            node = target.name
            self._save_cursor(_Cursor(next=node))

    # --- running ---------------------------------------------------------------------

    def get_steps(self) -> list[str]:
        return list(self._runners)

    async def run_step(self, step_name: str) -> Any:
        """Run one step on its own, against whatever is currently persisted. Valid in an
        edge-driven stage too — there's only ever one value per step, so "which pass" never
        comes up. A single-step run does not advance the cursor; it's a manual operation."""
        runner = self._runners.get(step_name)
        if runner is None:
            raise ValueError(f"Unknown step: {step_name}")
        return await runner()

    async def run_steps(self, *, fail_fast: bool = False, start_at: str | None = None) -> list[Any]:
        """Run the whole stage: sequentially through `edges` if it declares them, else by
        its dependency DAG with maximum parallelism. Either way the return is each step's
        value in declaration order.

        `fail_fast` applies to the DAG model only — it cancels in-flight steps and re-raises
        on the first failure, where the default records it, skips its dependents, and lets
        independent branches finish. A sequential graph has nothing to run in parallel and
        nothing to salvage, so a failing step always propagates, leaving the cursor on it so
        a re-run resumes there.

        `start_at` enters the graph at that step instead of at `START` or the saved cursor,
        then follows the edges as usual. Edge-driven stages only — a DAG has no single
        thread of control to drop into.
        """
        if start_at is not None:
            if self._graph is None:
                raise ValueError(
                    f"{self.stage_name}: start_at needs a stage that declares edges; this one "
                    f"runs as a dependency DAG, where a step has no single 'next'."
                )
            if start_at not in self._runners:
                raise ValueError(f"{self.stage_name}: unknown step {start_at!r}.")
        _LOGGER.info(format_module_start(module_name=self.stage_name, step_count=len(self._runners)))
        started = perf_counter()
        try:
            with ExitStack() as stack:
                # Fan out to every hook's stage() (enter in order, exit in reverse).
                for hook in self._hooks:
                    stack.enter_context(
                        hook.stage(stage_name=self.stage_name, step_count=len(self._runners))
                    )
                if self._graph is not None:
                    return await self._run_graph(start_at)
                assert self._scheduler is not None   # one model or the other is always built
                # The scheduler owns the loop; we hand it run_step (how to run one by name).
                return await self._scheduler.run(self.run_step, fail_fast=fail_fast)
        finally:
            elapsed_ms = int((perf_counter() - started) * 1000)
            _LOGGER.info(format_module_end(module_name=self.stage_name, elapsed_ms=elapsed_ms))
