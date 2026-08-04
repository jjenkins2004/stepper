"""One end-to-end run of the shape the framework exists for: a shared prefix, then the
same flow class mounted twice over different data, with a retry loop inside each mount."""

from pydantic import BaseModel

from stepper import EXIT, START, Flow, depends, edge, optional_depends, require, step


class Product(BaseModel):
    sku: str
    cents: int


class Channel(BaseModel):
    name: str
    markup: float
    rejects_first: bool = False


class Listing(BaseModel):
    sku: str
    cents: int
    channel: str
    retry_once: bool


class Receipt(BaseModel):
    channel: str
    tries: int
    ok: bool


class Report(BaseModel):
    lines: list[str]


class UploadFlow(Flow[Receipt]):
    listing = require(Listing)

    @step
    async def attempt(
        self, item=depends(listing), prev: Receipt | None = optional_depends("check")
    ) -> Receipt:
        tries = (prev.tries if prev else 0) + 1
        return Receipt(channel=item.channel, tries=tries, ok=False)

    @step
    async def check(self, tried=depends(attempt), item=depends(listing)) -> Receipt:
        ok = tried.tries > 1 or not item.retry_once
        return Receipt(channel=tried.channel, tries=tried.tries, ok=ok)

    edges = (
        edge(START).to(attempt),
        edge(attempt).to(check),
        edge(check).when(lambda r: r.ok).to(EXIT).otherwise(attempt),
    )


class ChannelFlow(Flow[Receipt]):
    product = require(Product)
    channel = require(Channel)

    @step
    async def listing(self, item=depends(product), spec=depends(channel)) -> Listing:
        return Listing(
            sku=item.sku,
            cents=round(item.cents * spec.markup),
            channel=spec.name,
            retry_once=spec.rejects_first,
        )

    upload = UploadFlow.bind(listing=listing)

    edges = (edge(START).to(listing), edge(listing).to(upload), edge(upload).to(EXIT))


class PushFlow(Flow[Report]):
    @step
    async def fetch(self) -> Product:
        return Product(sku="A1", cents=1000)

    @step
    async def tiktok_spec(self) -> Channel:
        return Channel(name="tiktok", markup=1.0)

    tiktok = ChannelFlow.bind(product=fetch, channel=tiktok_spec)

    @step
    async def shein_spec(self) -> Channel:
        return Channel(name="shein", markup=2.0, rejects_first=True)

    shein = ChannelFlow.bind(product=fetch, channel=shein_spec)

    @step
    async def report(self, t=depends(tiktok), s=depends(shein)) -> Report:
        return Report(lines=[f"{r.channel}:{r.tries}:{r.ok}" for r in (t, s)])

    edges = (
        edge(START).to(fetch),
        edge(fetch).to(tiktok_spec),
        edge(tiktok_spec).to(tiktok),
        edge(tiktok).to(shein_spec),
        edge(shein_spec).to(shein),
        edge(shein).to(report),
        edge(report).to(EXIT),
    )


def test_both_channels_publish_and_only_one_retries(run, mem):
    assert run(PushFlow().run(persist_service=mem)) == Report(
        lines=["tiktok:1:True", "shein:2:True"]
    )


def test_each_mount_keeps_its_own_namespace(run, mem):
    run(PushFlow().run(persist_service=mem))
    assert mem.fetch("push/tiktok/listing", Listing).cents == 1000
    assert mem.fetch("push/shein/listing", Listing).cents == 2000
    assert mem.fetch("push/tiktok/upload", Receipt).tries == 1
    assert mem.fetch("push/shein/upload", Receipt).tries == 2


def test_each_mounts_loop_checkpoints_separately(run, mem):
    run(PushFlow().run(persist_service=mem))
    assert "push/tiktok/upload/_loop_cursor" in mem._store
    assert "push/shein/upload/_loop_cursor" in mem._store


def test_a_branch_can_be_rerun_on_its_own(run, mem):
    """Re-running one branch runs it against what the last run left: its loop's back edge
    reads the previous `check`, so the attempt count carries on rather than restarting."""
    root = PushFlow().mount(persist_service=mem)
    run(root.run())
    assert run(root.run("shein")) == Receipt(channel="shein", tries=3, ok=True)


def test_the_shared_prefix_is_read_by_both_mounts(run, mem):
    root = PushFlow().mount(persist_service=mem)
    tiktok = root.node("tiktok")
    shein = root.node("shein")
    assert isinstance(tiktok, Flow) and isinstance(shein, Flow)
    reqs = type(tiktok).plan().requirements
    assert tiktok.key_of(reqs["product"]) == shein.key_of(reqs["product"]) == "push/fetch"
    assert tiktok.key_of(reqs["channel"]) == "push/tiktok_spec"
    assert shein.key_of(reqs["channel"]) == "push/shein_spec"
