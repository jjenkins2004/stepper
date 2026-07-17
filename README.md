# stepper

A tiny dependency-scheduled pipeline framework. Declare steps with `@step`, wire
their inputs with `depends`, group them into a `Stage`, and run a `Pipeline`. Run
order comes from the dependency graph — independent steps run concurrently — and each
step's return value is persisted so later steps (even in later stages) read it back
off disk.

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

asyncio.run(pipeline.run(module="all"))
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
  stage (a scheduling edge) or another stage (a disk input from an earlier stage).
- **`Stage`** lists its steps in `steps = (...)` — membership, *not* order. Run order
  is derived from `depends()` and validated at class creation (an unknown target or a
  cycle raises).
- **`Pipeline`** namespaces persistence by `output_root/name` (plus `/run_id` when given) and runs its
  stages. `run(module="all")` runs everything; `module=<stage>` runs one stage,
  `module=<stage>, step=<step>` runs one step.

## Persistence

`persist(key, value, model)` / `fetch(key, model)` store and reload a step's value through
the backend, which owns how it's encoded and where it lands — the only contract is that the
value round-trips. The default `DiskPersistService` writes one file per key: a `str` as
`.txt`, raw `bytes` under the key verbatim, anything else as `.json` (round-trips
int/list/BaseModel/etc.). A value can also be a `Persistable` — a model that runs its own
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
| `hooks` | `Pipeline(...)` / `Stage(...)` | `NoOpHooks()` | Context-manager hooks wrapping each step and stage — add tracing/metrics/actions with no framework tracing dep. |
| `configure_logging(level=, fmt=)` | top-level fn | `INFO`, `"%(message)s"` | Optional stdlib logging setup so `[STEP_*]` / `[MODULE_*]` lines print. |

## Public API

`Pipeline`, `StageFactory`, `Stage`, `Step`, `step`, `depends`, `Scheduler`,
`PersistService`, `DiskPersistService`, `Persistable`, `Hooks`, `NoOpHooks`,
`StepReport`, `configure_logging`.

## License

MIT — see [LICENSE](LICENSE).
