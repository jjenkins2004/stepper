# stepper

A tiny pipeline framework built on one idea: **a flow is a step made of steps.**

A step is an async function with declared inputs and one typed output. A flow is the same
shape — declared inputs, one typed output — except its output comes from the nodes inside
it, which are steps or other flows. That's the whole structure. There is no pipeline layer
above and no special case below.

**Why it's useful**

- **Nothing reaches into anything.** A step reads its own flow's steps, its own flow's
  children, or one of its flow's declared inputs. Nothing else is in scope, so wiring never
  crosses a flow boundary and nothing is ever resolved by name, proximity, or guesswork.
- **A flow is reusable because it's a function.** Declare what it needs with `require()`,
  bind that where you mount it, and mount it as many times as you like. Two mounts are two
  namespaces on disk, differing because they were called with different arguments.
- **Every value is on disk, at a path you can predict.** A node's path is the chain of
  names from the root down, and that path *is* its persist key. Read a run's output
  directory to see exactly what happened.
- **Errors happen when you write the class, not when you run it.** A flow's declaration is
  entirely about itself, so unbound inputs, mistyped bindings, unreachable nodes, a branch
  that can't finish, and a `depends()` that isn't guaranteed to have run all raise at
  import.
- **Loops that survive a crash.** A flow declares its control-flow graph, cycles included,
  checkpoints its way through, and resumes mid-loop where it died.
- **Parallelism is one edge with several targets.** `edge(a).to(b, c, d)` runs all three at
  once, `edge(b, c, d).to(m)` brings them back, and `m` is the node that reads every one of
  them.

Pure stdlib + pydantic — the framework depends on no tracing library. Add spans, metrics,
or any before/after action yourself via `Hooks` (see below).

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

from stepper import EXIT, START, Flow, depends, edge, require, step


class Order(BaseModel):
    total: int


class TaxFlow(Flow[Order]):
    order = require(Order)                      # what this flow needs

    @step
    async def with_tax(self, o=depends(order)) -> Order:
        return Order(total=int(o.total * 1.2))

    edges = (edge(START).to(with_tax), edge(with_tax).to(EXIT))


class CheckoutFlow(Flow[str]):
    @step
    async def build(self) -> Order:
        return Order(total=100)

    taxed = TaxFlow.bind(order=build)           # mounting it is the call

    @step
    async def summary(self, o=depends(taxed)) -> str:
        return f"order total: {o.total}"

    edges = (
        edge(START).to(build),
        edge(build).to(taxed),
        edge(taxed).to(summary),
        edge(summary).to(EXIT),
    )


asyncio.run(CheckoutFlow().run(run_id="run-1"))
```

Returns `"order total: 120"` and writes:

```
output/run-1/checkout.txt                     the root's own value
output/run-1/checkout/build.json              Order(total=100)
output/run-1/checkout/taxed/with_tax.json     Order(total=120)
output/run-1/checkout/taxed.json              the nested flow's value
output/run-1/checkout/summary.txt             "order total: 120"
output/run-1/checkout/_loop_cursor.json       where each graph got to
output/run-1/checkout/taxed/_loop_cursor.json
```

Every node's path is its key: `checkout/taxed/with_tax` is the step, `checkout/taxed` is the
flow it lives in, and both are on disk.

## Core concepts

### Producers

Three things have an output, and `depends()` takes any of them — a step consuming one
can't tell which it got:

| | what it is |
|---|---|
| `@step` | an output it computes itself |
| `SomeFlow.bind(...)` | an output its subtree computes |
| `require(Model)` | an output with no producer, supplied by whoever mounts this flow |

All three are **class attributes**, and that isn't stylistic: a `depends()` is a parameter
default, evaluated while the class body runs. There is no `self` at that moment and never
will be, so anything a step reads has to already be an attribute. That's also what confines
wiring to one class — every `depends()` names something declared right there.

### `require()` and `.bind()`

A flow is a function. `require()` declares its parameters, `Flow[Model]` declares its
return type, and mounting it is the call:

```python
class ChannelFlow(Flow[Receipt]):
    product = require(Product)
    channel = require(Channel)
    ...

tiktok = ChannelFlow.bind(product=normalize, channel=tiktok_spec)
shein  = ChannelFlow.bind(product=normalize, channel=shein_spec)
```

Each keyword is one of the child's requirements; each value is a producer declared on the
mounting class. Both are checked when that class is created.

Two mounts of one class collide with nothing, because a binding names both ends. There is
no resolution step to get wrong: `push/tiktok/upload` and `push/shein/upload` are different
keys because they're different nodes.

**Closed and open.** A flow with no requirements is *closed* — it runs on its own,
anywhere. One with requirements is *open*: it runs wherever they're supplied. To run an open
flow by itself, write a small closed flow that supplies them. The harness is a flow like any
other, so its inputs persist and the standalone run is as inspectable as a real one.

```python
class TiktokOnlyFlow(Flow[Receipt]):
    @step
    async def product(self) -> Product: ...
    @step
    async def channel(self) -> Channel: ...

    only = ChannelFlow.bind(product=product, channel=channel)

    edges = (edge(START).to(product), edge(product).to(channel),
             edge(channel).to(only), edge(only).to(EXIT))
```

### `output`

A flow's value is whatever node it *ended* on. The nodes with an edge to `EXIT` are its
terminals, and they all have to produce the same model — `Flow[Receipt]` is the contract,
checked against them when the class is created.

The flow persists that value under its own path, so `depends(some_flow)` reads a key like
any other, and a finished subtree leaves one artifact naming it.

### Paths

A node's path is the `/`-joined chain of names from the root down, root included — the
attribute name each node was declared under, all the way up. It is also the key it persists
under. `run_id` prefixes *keys* rather than paths, so two runs never collide on the backend
while an address means the same thing in every run.

```python
await root.run()                        # everything
await root.run("tiktok")                # that subtree
await root.run("tiktok/upload/attempt") # one step, against what's persisted
```

Each node eats the first segment and hands the rest down, so `run("tiktok/upload/attempt")`
is `tiktok` asking `upload` to run `attempt`.

## Order

Every flow declares its whole graph, `START` to `EXIT`, and runs it sequentially: run a
node, ask its edge where to go, repeat. Nothing is inferred — `depends()` is dataflow, and
it's checked *against* the graph rather than defining it.

```python
class ChallengeFlow(Flow[Analysis]):
    @step
    async def edit(self, prev: Analysis | None = optional_depends("analysis")) -> Draft: ...

    @step
    async def audit(self, draft=depends(edit)) -> AuditResult: ...

    @step
    async def analysis(self, draft=depends(edit)) -> Analysis: ...

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
- **`.to(a, b, c)`** — several targets is a **fan-out**: they all run, concurrently, and
  control resumes where their arms meet. **`edge(a, b, c).to(join)`** is the way back in:
  one edge declared once per source, unconditional. See below.
- **`edge(START).to(first)`** — where the graph begins. Exactly one, unconditional (there's
  no result to branch on yet). **`EXIT`** ends it.
- **Sources and targets** may be steps or nested flows. A predicate on either gets that
  node's typed output — `edge(audit)` is an `Edge[AuditResult]`, so `r.passd` is an editor
  error. Predicates are only ever *called*: nothing reads a return type to pick a route.
  They must be pure (resume re-evaluates them: no clock, no randomness, no network) and see
  their source's output alone. A decision needing another fact reads it via `depends()`, or
  the source puts it in its own output model.
- **`max_steps`** (`Flow` ClassVar, default 1000) — runaway fuse; blowing it raises. One
  budget for the whole graph, concurrent arms included. Reaching `EXIT` is how a graph is
  meant to end, so set this well above any real run.

Declaring `edges` *below* the steps is what makes a back edge legal, and `depends("name")`
is how a step reads a producer declared below it (a name carries no type, so annotate the
parameter yourself, as `edit` does above).

The framework tracks no round number and no loop state. Counters live in a step's own output
model, where they're typed and persisted — `r.rounds` above is a field a step wrote.

### Fan out, fan back in

Give one `.to(...)` several targets and they all run, at the same time:

```python
class ListingFlow(Flow[Listing]):
    @step
    async def brief(self) -> Brief: ...

    @step
    async def images(self, b=depends(brief)) -> Images: ...
    @step
    async def copy(self, b=depends(brief)) -> Copy: ...
    @step
    async def pricing(self, b=depends(brief)) -> Pricing: ...

    @step
    async def assemble(self, i=depends(images), c=depends(copy), p=depends(pricing)) -> Listing: ...

    edges = (
        edge(START).to(brief),
        edge(brief).to(images, copy, pricing),         # fans out
        edge(images, copy, pricing).to(assemble),      # and back together
        edge(assemble).to(EXIT),
    )
```

Several *targets* fan out; several *sources* is the shorthand for coming back in, since
that's three edges you'd write anyway. A multi-source edge routes unconditionally — its
sources have no one result to branch on — and it's sugar and nothing else, so `edge(images)`
elsewhere in the same tuple is the same "one edge per node" error it has always been.

The node the arms meet at — `assemble` — is where the flow carries on, and it is the one
node that may `depends()` on all of them: every arm ran, so every arm's output is behind it.
You never name it. It's derived as the first node every arm is guaranteed to reach, which is
what lets an arm branch, loop, or fan out again without changing how a fan-out is written:

```python
edge(check).when(lambda r: r.heavy).to(images, copy).otherwise(reuse)   # this branch fans out
edge(images).to(crop, tag)                                              # and again, inside it
```

Arms are independent. They run concurrently, so a `depends()` from one arm into another is a
race, not an ordering, and it's rejected — the join is where arms are read. Nothing may jump
into the middle of an arm, and no arm may loop back to the node that fanned out; loop from
the join instead. All of it is checked when the class is created.

Arms whose only meeting point is `EXIT` are rejected too. A flow's value is whatever node it
*ended* on, and several arms ending at once names no node — so `edge(a).to(b, c)` with `b`
and `c` both routing to `EXIT` is a declaration error, not a coin flip. Add the node that
reads them.

A step is `async`, so an arm that's pure CPU blocks the others; hand that to a thread or a
process yourself, as you would anywhere else. Hooks see arms concurrently — several `step()`
context managers open at once, closing in whatever order the arms finish.

### Checked when the class is created

Every node must have an edge out, be reachable from `START`, and be able to reach `EXIT`.
That last one is per-node, not "there's an `EXIT` somewhere": a branch arm that drops into
an inescapable sub-cycle is rejected. Plus exactly one unconditional single-target
`edge(START)`, no node named by two `edge(...)`s, `.otherwise` on every branch, and targets
that are nodes of this flow. A multi-source edge is unconditional and single-target: it fans
in, so it can't also branch or fan out.

Every fan-out must have somewhere its arms meet, and that place must be a node rather than
`EXIT`. Its arms must be disjoint, and none of them may route back to the node that fanned
out. Above all, an arm is entered *only* by the fan-out taking it — nothing else routes into
one, the graph doesn't begin inside one, and the source itself may not reach one by another
branch. That last rule is the subtle one, and it's what the join's `depends()` rests on:

```python
edge(check).when(...).to(extra, common).otherwise(common)   # rejected
```

`common` runs either way, so it reads as safe — but on the `.otherwise` route it runs
*alone*, and a join that depends on `extra` would find nothing. Give the other branch its
own node.

Dependencies are checked against the graph too:

| wiring | must be satisfiable |
|---|---|
| `depends()` | on **every** path from `START` — else the fetch finds nothing, every time |
| `optional_depends()` | on **at least one** path — else it's None forever and the wiring is dead |
| across two arms of a fan-out | never — they run at once, so it's a race; read them at the join |
| a producer that persists nothing | exempt — there's no fetch to fail |

A fan-out's join is the exception to "every path": all of its arms ran, so everything they
produced counts as available there, which is what makes `assemble` above legal.

That split is what makes a back edge legal: in `START → b`, `b → a`, `a → b`, the second
visit to `b` has `a` behind it even though the first doesn't.

And, on top of the graph: a `depends()` on another flow's node, a binding to a producer
declared elsewhere, a binding whose model doesn't match the `require()`, an unbound
requirement, a name bound to two producers, and terminals that disagree about what the flow
produces.

One more, from the persistence side: a step whose return model is a plain `BaseModel`
holding a `Persistable` — at any depth, including through a union or a container. The hooks
run only for the model `persist` is *given*, so a buried one would have its fields dumped as
JSON and its side-artifacts dropped: the metadata reads back intact and the blobs are simply
gone. Return the `Persistable` itself, or make the outer model a `Persistable` whose
`on_persist`/`on_fetch` forward to it. Checking a step's return type covers every model that
crosses the persist layer, since a flow's output must equal a terminal's and a `require()`
must match the producer bound to it.

### Passes and resume

A node overwrites its key on every pass, so `depends()` always reads the latest — right
either way, since an upstream in the same pass has already written it and a cross-pass read
wants the previous one. Nothing accumulates: no retention knob, no "which pass" question.

Progress is checkpointed to `<flow path>/_loop_cursor` — the next node's name, nothing else
— so a crashed run resumes there instead of at `START`. A fan-out has no single "next", so
while its arms are in flight the cursor names the node that fanned out: a crash there
resumes by re-running it and every arm. Checkpoint writes are best-effort. A
cursor that's missing, corrupt, or naming a node the graph no longer has restarts from
`START`; a backend that can't be read *at all* propagates, since restarting silently would
repeat side effects.

Three ways to run, all through the one `run`:

| call | what happens |
|---|---|
| `run()` | from `START`, or from the cursor if a previous run crashed |
| `run("upload/attempt")` | that node alone, against what's persisted; cursor untouched |
| `run("upload/attempt", follow_edges=True)` | enter `upload`'s graph at `attempt` and follow edges to `EXIT` |

`follow_edges` overrides a saved cursor — an explicit entry beats an automatic one — and
checkpoints as it goes, so that run is itself resumable.

> **Resume restores values, not side effects.** The node a run restarts at may have already
> half-run, so it runs again in full — nodes must be re-entrant. Anything the framework
> can't undo (a spawned job, a written workspace) is yours to make safe. A resumed run also
> leaves mixed-generation state on disk: nodes that ran this pass hold current values, nodes
> that didn't still hold the previous pass's. The same applies to re-running a finished flow
> without a fresh `run_id` — it starts at `START`, but against whatever the last run left.

## Persistence

`persist(key, value, model)` / `fetch(key, model)` store and reload a value through the
backend, which owns how it's encoded and where it lands — the only contract is that the
value round-trips. The default `DiskPersistService` writes one file per key: a `str` as
`.txt`, raw `bytes` under the key verbatim, anything else as `.json` (round-trips
int/list/BaseModel/etc.). `InMemoryPersistService` is the same encoding into a dict instead
of files — no disk at all, for a run whose output must not be written anywhere.

`base_dir` is the backend's business, not a flow's; a run's separation comes from the
`run_id` the root bakes into every key.

A value can also be a `Persistable` — a `BaseModel` that hooks into the persist/fetch
lifecycle. Its plain fields still serialize as JSON metadata; on top of that,
`on_persist`/`on_fetch` let the model persist and reload anything else it owns (large blobs,
derived artifacts, external references) by calling `persist`/`fetch` again with its own keys
— usually `bytes` under a sub-key like `f"{key}/image.png"`. The key is opaque, so bake any
backend naming (a file extension, a bucket path) into it. `persist` writes the fields, then
calls `on_persist(service, key)`; `fetch` rebuilds the model, then calls `on_fetch(service,
key)` so it can stash the service+key and lazy-load later. Keep the extra state in
`PrivateAttr` so the metadata dump skips it. Consumers that need images (and PIL) build
their own `Persistable` — stepper stays pydantic-only.

## Telemetry / hooks

The framework depends on no tracing library. To add spans, metrics, or any before/after
action, pass a `Hooks` implementation to the flow you run. Each hook is a context manager
wrapped around the work — code before `yield` runs before the node, code after runs when it
finishes or raises:

```python
from contextlib import contextmanager

import logfire

from stepper import StepReport


class LogfireHooks:
    @contextmanager
    def step(self, *, path, input_type, output_type):
        report = StepReport()
        with logfire.span("step {path}", path=path,
                          input_type=input_type, output_type=output_type) as span:
            yield report                          # framework fills report after the step runs
            if report.has_output:
                span.set_attribute("output", report.output)

    @contextmanager
    def flow(self, *, path, node_count):
        with logfire.span("flow {path}", path=path, node_count=node_count):
            yield


await MyFlow().run(hooks=LogfireHooks())
```

A node is identified by its `path`, which is also its persist key, so a span and its
artifact carry the same name.

**Running several hooks.** Pass a list — `hooks=[LogfireHooks(), StreamHooks(reducer)]` —
and stepper fans out natively: no composite wrapper. Each is entered before the node in list
order and exited in reverse, an exception propagates into every one at its `yield`, and each
gets the step output in its own `StepReport`. A lone `hooks=SomeHooks()` behaves exactly as
one hook always has.

**Capturing a step's output.** The output only exists *after* the step runs (after your
`yield`), so you can't yield it. Instead yield a `StepReport`: the framework calls
`report.set_output(result)` once the step has run and persisted, and your after-`yield` code
reads `report.output` (guard with `report.has_output` — a step with no return annotation
persists nothing and leaves the report empty). You never fill it; you never implement a
method — the framework only ever touches its own `StepReport` type, so there's no tracing
coupling. Yield nothing if you don't need the output.

Exactly when each runs:

- **`step(...)`** — code before `yield` runs **before** the step's inputs are fetched and
  its body runs; code after `yield` runs **after** the body returns *and* its output is
  persisted. If the step raises, the after-`yield` code is **skipped** and the exception
  propagates through your context manager — use `try/except` (or `try/finally`) if you need
  to observe failures.
- **`flow(...)`** — before-`yield` runs before any node in the flow starts; after-`yield`
  runs once the flow finishes.

## Configuration

| Knob | Where | Default | What it does |
|---|---|---|---|
| `run_id` | `run(...)` / `mount(...)` | `None` | Prefixes every persist key, so repeat runs don't overwrite each other. Node paths are unaffected, so an address means the same thing in every run. |
| `persist_service` | `run(...)` / `mount(...)` | `DiskPersistService(base_dir="output")` | The backend, which owns where output physically lands. |
| `hooks` | `run(...)` / `mount(...)` | `()` | One `Hooks` or a list of them, wrapping every node in the tree. Several fan out: entered in order, exited in reverse. |
| `edges` | `Flow` class body | — | Required. The flow's whole control-flow graph, `START` to `EXIT`. |
| `max_steps` | `Flow` class body | `1000` | Runaway fuse: raises after this many node executions without reaching `EXIT`. One budget for the whole graph, concurrent arms included. |
| `configure_logging(level=, fmt=)` | top-level fn | `INFO`, `"%(message)s"` | Optional stdlib logging setup so `[STEP_*]` / `[MODULE_*]` lines print. |

## Public API

`Flow`, `FlowRef`, `Node`, `Step`, `step`, `Producer`, `Requirement`, `require`, `depends`,
`optional_depends`, `edge`, `Edge`, `START`, `EXIT`, `PersistService`, `DiskPersistService`,
`InMemoryPersistService`, `Persistable`, `Hooks`, `StepReport`, `configure_logging`.

`tests/test_integration.py` is a runnable end-to-end example: one product, two
marketplaces, one `ChannelFlow` class mounted twice, with a retry loop inside each mount.

## License

MIT — see [LICENSE](LICENSE).
