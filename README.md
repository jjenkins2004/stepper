# stepper

A tiny pipeline framework. Declare steps with `@step`, wire their inputs with `depends`,
group them into a `Stage`, run a `Pipeline`. Run order comes from the wiring, every step's
return value is persisted, and a stage can loop.

**Why it's useful**

- **Order you never maintain.** `depends()` says what a step *reads*; run order and
  concurrency fall out of it. Insert or reorder a step and nothing else changes — and an
  unknown target or a cycle raises when the class is created, not mid-run.
- **Every step's output is on disk.** Read a run's output dir to see what happened, re-run
  one step against what's already there, or feed a later stage from an earlier one. Debug
  by inspection instead of by re-running the world.
- **Wiring is typed.** `depends(build_order)` gives the consuming parameter the producer's
  return type, so a mismatch is an editor error.
- **Loops that survive a crash.** A stage can declare a control-flow graph with cycles and
  branches, checkpoint its way through, and resume mid-loop where it died.

Pure stdlib + pydantic — the framework depends on no tracing library. Add spans,
metrics, or any before/after action yourself via `Hooks` (see below).

> Name note: the `stepper` name on PyPI belongs to an unrelated stepper-motor library.
> Install this straight from git (below); it is never published to PyPI.

## Install

```bash
pip install git+https://github.com/jjenkins2004/stepper.git
```

## Quickstart

```python
import asyncio

from pydantic import BaseModel

from stepper import Pipeline, Stage, depends, step


class Order(BaseModel):
    total: int


class ExtractStage(Stage):
    @step
    async def build_order(self) -> Order:
        return Order(total=100)

    steps = (build_order,)


class ReportStage(Stage):
    @step
    async def summary(self, order=depends(ExtractStage.build_order)) -> str:
        return f"order total: {order.total}"

    steps = (summary,)


pipeline = Pipeline(
    name="orders",
    run_id="run-1",
    output_root="output",  # writes output/orders/run-1/<Stage>/<step>.{json,txt}
    stages={
        "extract": lambda ps: ExtractStage(persist_service=ps),
        "report": lambda ps: ReportStage(persist_service=ps),
    },
)

asyncio.run(pipeline.run(stage="all"))
```

This writes `output/orders/run-1/Extract/build_order.json` and
`output/orders/run-1/Report/summary.txt`. `depends(ExtractStage.build_order)` fetches
the persisted `Order` and passes it into `summary`.

## Core concepts

- **`@step`** turns an async `Stage` method into a step. Its return annotation is the
  model persisted/fetched for it — a `Persistable` also stores its own side-artifacts (on
  the default disk backend: `str` → `.txt`, everything else → `.json`). No return
  annotation ⇒ nothing is persisted.
- **`depends(producer)`** wires a parameter to another step's persisted output — same
  stage (a scheduling edge) or another stage (a disk input from an earlier stage). A step
  *name* works too, resolved at class creation, for a producer defined further down the
  class body (see [Loops](#loops)). A producer that persists nothing is pure ordering: the
  consumer still runs after it, and the parameter is `None` — typed that way, since such a
  step is a `Step[None]`.
- **`optional_depends(producer)`** is `depends` typed `R | None`: if the producer's value
  isn't persisted when this step runs, the parameter is None instead of the run raising.
  Scheduling is unchanged — a same-stage producer is still waited for and, if it *fails*,
  this step is still skipped; only a *missing* persisted value becomes None.
- **`Stage`** lists its steps in `steps = (...)` — membership, *not* order. Run order
  is derived from `depends()` and validated at class creation (an unknown target or a
  cycle raises).
- **`Pipeline`** namespaces persistence by `output_root/name` (plus `/run_id` when given) and runs its
  stages. `run(stage="all")` runs everything; `stage=<name>` runs one stage,
  `stage=<name>, step=<step>` runs one step, and adding `follow_edges=True` enters an
  edge-driven stage's graph at that step and keeps going (see [Loops](#loops)). `run`
  returns the last thing it ran — a single step's value, or the final stage's step
  results — so callers can read the final output without going back to the
  `PersistService`.

## Loops

A stage orders itself one of two ways, never both:

- **No `edges`** — `depends()` is the order; independent steps run concurrently. Everything
  above, unchanged.
- **`edges = (...)`** — you declare the whole graph and it runs sequentially, `START` to
  `EXIT`. Nothing is inferred.

The second is how a stage loops, since `depends()` is dataflow and can't express a cycle
or a branch:

```python
from stepper import EXIT, START, Stage, depends, edge, optional_depends, step


class ChallengeStage(Stage):
    @step
    async def edit(self, prev: Analysis | None = optional_depends("analysis")) -> Draft: ...

    @step
    async def audit(self, draft=depends(edit)) -> AuditResult: ...

    @step
    async def analysis(self, draft=depends(edit)) -> Analysis: ...

    steps = (edit, audit, analysis)

    edges = (
        edge(START).to(edit),
        edge(edit).to(audit),
        edge(audit).when(lambda r: r.passed).to(EXIT).otherwise(analysis),
        edge(analysis)
            .when(lambda r: r.rounds >= 8).to(EXIT)
            .otherwise(edit),
    )
```

### The interface

- **`edge(source)`** — unconditional `.to(target)`, or a branch chain
  `.when(pred).to(target)…` closed by `.otherwise(target)`. First matching predicate wins;
  `.otherwise` is required, so exhaustiveness is structural rather than hoped for.
- **`edge(START).to(first)`** — where the graph begins. Exactly one, unconditional (there's
  no result to branch on yet). **`EXIT`** ends it.
- **Predicates** take the source step's result and return a bool. Typed — `edge(audit)` is
  an `Edge[AuditResult]`, so `r.passd` is an editor error — and only ever *called*: nothing
  reads a return type to pick a route. They must be pure (resume re-evaluates them: no
  clock, no randomness, no network) and they see their source's output alone. A decision
  needing another step's fact declares a `depends()` for it, or the source puts it in its
  own output model.
- **`max_steps`** (`Stage` ClassVar, default 1000) — runaway fuse; blowing it raises.
  Reaching `EXIT` is how a graph is meant to end, so set this well above any real run.

Steps don't change: inputs via `depends()`, one output model, no routing tokens in the
return annotation. In an edge-driven stage `depends()` is data wiring only — the edges own
ordering. Declaring `edges` *below* the steps is what makes the back edge legal, and
`depends("name")` is how a step reads a producer defined below it (a name carries no type,
so annotate the parameter yourself, as `edit` does above).

The framework tracks no round number and no loop state. Counters live in the step's own
output model, where they're typed and persisted — `r.rounds` above is a field a step wrote.

### Checked at class creation

Every step must have an edge out, be reachable from `START`, and be able to reach `EXIT`.
That last one is per-step, not "there's an `EXIT` somewhere": a branch arm that drops into
an inescapable sub-cycle is rejected. Plus exactly one unconditional `edge(START)`, no
duplicate edges, `.otherwise` on every branch, and targets that are listed steps.

Dependencies are checked against the graph too:

| wiring | must be satisfiable |
|---|---|
| `depends()` | on **every** path from `START` — else the fetch finds nothing, every time |
| `optional_depends()` | on **at least one** path — else it's None forever and the wiring is dead |
| producer that persists nothing | exempt — there's no fetch to fail |

That split is what makes a back edge legal: in `START → b`, `b → a`, `a → b`, the second
visit to `b` has `a` behind it even though the first doesn't.

### Passes and resume

A step overwrites `<Stage>/<step>` on every pass, so `depends()` always reads the latest —
right either way, since an upstream in the same pass has already written it and a
cross-pass read wants the previous one. Nothing accumulates: no retention knob, no "which
pass" question.

Progress is checkpointed to `<Stage>/_loop_cursor` — the next step's name, nothing else —
so a crashed run resumes there instead of at `START`. Checkpoint writes are best-effort. A
cursor that's missing, corrupt, or naming a step the graph no longer has restarts from
`START`; a backend that can't be read *at all* propagates, since restarting silently would
repeat side effects.

Three ways to run an edge-driven stage, all through the one `run`:

| call | what happens |
|---|---|
| `run(stage="challenge")` | from `START`, or from the cursor if a previous run crashed |
| `run(stage="challenge", step="blind")` | that step alone, against what's persisted; cursor untouched |
| `run(stage="challenge", step="blind", follow_edges=True)` | enter the graph at `blind` and follow edges to `EXIT` |

`follow_edges` overrides a saved cursor — an explicit entry beats an automatic one — and
checkpoints as it goes, so that run is itself resumable. It needs a stage that declares
`edges`; a DAG has no single thread of control to drop into.

> **Resume restores values, not side effects.** The step a run restarts at may have
> already half-run, so it runs again in full — steps in an edge-driven stage must be
> re-entrant. Anything the framework can't undo (a spawned job, a written workspace) is
> yours to make safe. A resumed run also leaves mixed-generation state on disk: steps that
> ran this pass hold current values, steps that didn't still hold the previous pass's.

## Persistence

`persist(key, value, model)` / `fetch(key, model)` store and reload a step's value through
the backend, which owns how it's encoded and where it lands — the only contract is that the
value round-trips. The default `DiskPersistService` writes one file per key: a `str` as
`.txt`, raw `bytes` under the key verbatim, anything else as `.json` (round-trips
int/list/BaseModel/etc.). `InMemoryPersistService` is the same encoding into a dict instead
of files — no disk at all, for a single run whose output must not be written anywhere (pass
it as `persist_service=`). A value can also be a `Persistable` — a model that runs its own
persistence on top.

A `Persistable` is a `BaseModel` that hooks into the persist/fetch lifecycle. Its plain
fields still serialize as JSON metadata; on top of that, `on_persist`/`on_fetch` let the
model persist and reload anything else it owns (large blobs, derived artifacts, external
references) by calling `persist`/`fetch` again with its own keys — usually `bytes` under a
sub-key like `f"{key}/image.png"`. The key is opaque, so bake any backend naming (a file
extension, a bucket path) into it. `persist` writes the fields, then calls
`on_persist(service, key)`; `fetch` rebuilds the model, then calls `on_fetch(service, key)`
so it can stash the service+key and lazy-load later. Keep the extra state in `PrivateAttr`
so the metadata dump skips it. Consumers that need images (and PIL) build their own
`Persistable` — stepper stays pydantic-only.

## Telemetry / hooks

The framework depends on no tracing library. To add spans, metrics, or any
before/after action, pass a `Hooks` implementation to `Pipeline` (or a `Stage`). Each
hook is a context manager wrapped around the work — code before `yield` runs before
the step/stage, code after runs when it finishes or raises:

```python
from contextlib import contextmanager

import logfire

from stepper import Pipeline, StepReport


class LogfireHooks:
    @contextmanager
    def step(self, *, stage_name, step_name, input_type, output_type):
        report = StepReport()
        with logfire.span("step {step_name}", step_name=step_name, stage=stage_name,
                          input_type=input_type, output_type=output_type) as span:
            yield report                          # framework fills report after the step runs
            if report.has_output:
                span.set_attribute("output", report.output)

    @contextmanager
    def stage(self, *, stage_name, step_count):
        with logfire.span("stage {stage_name}", stage_name=stage_name, step_count=step_count):
            yield


pipeline = Pipeline(..., hooks=LogfireHooks())
```

**Running several hooks.** Pass a list — `hooks=[LogfireHooks(), StreamHooks(reducer)]`
(on a `Pipeline` or a `Stage`) — and stepper fans out natively: no composite wrapper.
Each is entered before the step/stage in list order and exited in reverse, an exception
propagates into every one at its `yield`, and each gets the step output in its own
`StepReport`. A lone `hooks=SomeHooks()` behaves exactly as one hook always has.

**Capturing a step's output.** The output only exists *after* the step runs (after your
`yield`), so you can't yield it. Instead yield a `StepReport`: the framework calls
`report.set_output(result)` once the step has run and persisted, and your after-`yield`
code reads `report.output` (guard with `report.has_output` — a step with no return
annotation persists nothing and leaves the report empty). You never fill it; you never
implement a method — the framework only ever touches its own `StepReport` type, so
there's no tracing coupling. Yield nothing if you don't need the output.

Exactly when each runs:

- **`step(...)`** — code before `yield` runs **before** the step's inputs are fetched
  and its body runs; code after `yield` runs **after** the body returns *and* its
  output is persisted. If the step raises, the after-`yield` code is **skipped** and
  the exception propagates through your context manager — use `try/except` (or
  `try/finally`) if you need to observe failures.
- **`stage(...)`** — before-`yield` runs before any step in the stage starts;
  after-`yield` runs once every step has finished.

The default (`NoOpHooks`) does nothing. Because tracing lives entirely in your hook,
the framework never sees `logfire` (or a `run_id` contextvar) — bake whatever context
you want into your hooks instance (e.g. `LogfireHooks(run_id=...)`).

## Configuration

| Knob | Where | Default | What it does |
|---|---|---|---|
| `output_root` | `Pipeline(...)` | `Path("output")` | Root dir for run output; final path is `output_root/name`, plus `/run_id` when `run_id` is given. Relative paths resolve against cwd. |
| `run_id` | `Pipeline(...)` | `None` | Optional per-run subdir under `output_root/name`, so separate runs don't clobber each other. Omit it and output lands directly in `output_root/name`. Ignored when you pass your own `persist_service`. |
| `persist_service` | `Pipeline(...)` | disk backend under `output_root` | Swap in any `PersistService` (e.g. in-memory or object store); wins over `output_root`. |
| `hooks` | `Pipeline(...)` / `Stage(...)` | `NoOpHooks()` | One `Hooks` or a list of them, wrapping each step and stage — add tracing/metrics/actions with no framework tracing dep. Several fan out: entered in order, exited in reverse. |
| `edges` | `Stage` class body | `()` | Declare the stage's whole control-flow graph, `START` to `EXIT`, and it runs sequentially instead of as a `depends()` DAG. That's how a stage loops. |
| `max_steps` | `Stage` class body | `1000` | Runaway fuse for an edge-driven stage: raises after this many step executions without reaching `EXIT`. |
| `fail_fast` | `Pipeline(...)` / `run_steps(...)` | `False` | `True`: a stage cancels its in-flight steps and re-raises on the first step failure. Default: record it, skip its dependents, let independent branches finish. A single-step run always re-raises. |
| `configure_logging(level=, fmt=)` | top-level fn | `INFO`, `"%(message)s"` | Optional stdlib logging setup so `[STEP_*]` / `[MODULE_*]` lines print. |

## Public API

`Pipeline`, `StageFactory`, `Stage`, `Step`, `step`, `depends`, `optional_depends`,
`edge`, `Edge`, `START`, `EXIT`, `Scheduler`,
`PersistService`, `DiskPersistService`, `InMemoryPersistService`, `Persistable`, `Hooks`,
`NoOpHooks`, `StepReport`, `configure_logging`.

## License

MIT — see [LICENSE](LICENSE).
