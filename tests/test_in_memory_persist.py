"""InMemoryPersistService: the disk round-trip contract, into a dict — no files.

No ``from __future__ import annotations`` — ``Step`` reads raw ``fn.__annotations__``, so
stringized return annotations would break ``.model`` inference for the stages below.

Mirrors test_persist.py minus the on-disk path assertions (there are no files): checks the
str/bytes/JSON round-trip, the missing-key ``FileNotFoundError`` (which drives optional
deps), Persistable metadata + blobs, that reads return independent copies (serialize on
write, not a stashed live object), and a full pipeline run with the backend injected —
asserting nothing lands on disk.
"""

import pytest
from pydantic import BaseModel, PrivateAttr

from stepper import (
    InMemoryPersistService,
    Persistable,
    PersistService,
    Pipeline,
    Stage,
    depends,
    optional_depends,
    step,
)


class Item(BaseModel):
    value: int


class Bag(BaseModel):
    tags: list[str]


class Doc(Persistable):
    """Toy Persistable (mirrors test_persist.py): JSON metadata (``name`` + ``ids``) plus a
    PrivateAttr blob map persisted as ``bytes`` under sub-keys and lazy-read after fetch."""

    name: str
    ids: list[str]
    _blobs: dict[str, bytes] = PrivateAttr(default_factory=dict)
    _service: "PersistService | None" = PrivateAttr(default=None)
    _key: str = PrivateAttr(default="")

    def on_persist(self, service, key):
        for id in self.ids:
            service.persist(f"{key}/{id}.bin", self._blobs[id], bytes)  # extension baked into key

    def on_fetch(self, service, key):
        self._service = service
        self._key = key

    def load(self, id):  # lazy: reads the blob off the service on demand
        assert self._service is not None
        return self._service.fetch(f"{self._key}/{id}.bin", bytes)


# --- round-trip: str / bytes / model / int / list ----------------------------------


def test_round_trips_str_bytes_and_models():
    svc = InMemoryPersistService()
    svc.persist("Extract/text", "hello", str)
    svc.persist("Render/thumb.png", b"\x00\x01\x02", bytes)
    svc.persist("Build/item", Item(value=7), Item)
    svc.persist("Build/count", 3, int)
    svc.persist("Build/ids", ["a", "b"], list[str])

    assert svc.fetch("Extract/text", str) == "hello"
    assert svc.fetch("Render/thumb.png", bytes) == b"\x00\x01\x02"
    assert svc.fetch("Build/item", Item) == Item(value=7)
    assert svc.fetch("Build/count", int) == 3
    assert svc.fetch("Build/ids", list[str]) == ["a", "b"]


def test_fetch_missing_raises_file_not_found():
    # FileNotFoundError (not KeyError) so the Stage runner turns a missing *optional* dep
    # into None instead of raising (see test_optional_dep_missing_reads_as_none).
    svc = InMemoryPersistService()
    with pytest.raises(FileNotFoundError, match="Render/thumb"):
        svc.fetch("Render/thumb.png", bytes)


# --- serialize-on-write: reads are independent copies ------------------------------


def test_read_returns_independent_copy():
    svc = InMemoryPersistService()
    item = Item(value=1)
    svc.persist("Build/item", item, Item)

    item.value = 99                                   # mutate the original after persist
    assert svc.fetch("Build/item", Item).value == 1   # the store held a snapshot

    a = svc.fetch("Build/item", Item)
    b = svc.fetch("Build/item", Item)
    assert a is not b                                 # each read decodes a fresh object
    a.value = 42
    assert svc.fetch("Build/item", Item).value == 1   # mutating a copy can't leak back


def test_read_deep_copies_nested_fields():
    # Not just a flat scalar: a nested mutable field must be independent per read too,
    # matching disk's JSON round-trip — a shallow model_copy would leak this in-place edit.
    svc = InMemoryPersistService()
    svc.persist("Build/bag", Bag(tags=["a"]), Bag)

    a = svc.fetch("Build/bag", Bag)
    a.tags.append("b")                                # mutate the nested list in place
    assert svc.fetch("Build/bag", Bag).tags == ["a"]  # store's snapshot untouched


# --- Persistable metadata + blobs --------------------------------------------------


def test_persistable_round_trips_metadata_and_blobs():
    svc = InMemoryPersistService()
    doc = Doc(name="report", ids=["a", "b"])
    doc._blobs = {"a": b"alpha", "b": b"beta"}
    svc.persist("Build/doc", doc, Doc)

    back = svc.fetch("Build/doc", Doc)
    assert back.name == "report"
    assert back.ids == ["a", "b"]
    assert not back._blobs           # blobs live under sub-keys, not the metadata
    assert back.load("a") == b"alpha"
    assert back.load("b") == b"beta"


# --- injected into a Pipeline: runs, and writes nothing to disk --------------------


class ProduceStage(Stage):
    @step
    async def doc(self) -> Doc:
        d = Doc(name="report", ids=["a"])
        d._blobs = {"a": b"payload"}
        return d

    steps = (doc,)


class ConsumeStage(Stage):
    @step
    async def read(self, doc=depends(ProduceStage.doc)) -> str:
        return f"{doc.name}:{doc.load('a').decode()}"

    steps = (read,)


def test_injected_backend_runs_pipeline_without_touching_disk(tmp_path, run):
    svc = InMemoryPersistService()
    p = Pipeline(
        name="p",
        run_id="r1",
        output_root=tmp_path,     # ignored: an explicit persist_service wins
        persist_service=svc,
        stages={
            "produce": lambda ps: ProduceStage(persist_service=ps),
            "consume": lambda ps: ConsumeStage(persist_service=ps),
        },
    )
    run(p.run(module="all"))

    assert p.persist_service is svc                       # injected backend won
    assert svc.fetch("Consume/read", str) == "report:payload"
    assert list(tmp_path.iterdir()) == []                # the point: nothing on disk


# --- optional dep: a missing value reads back as None (the FileNotFoundError path) --


class MaybeStage(Stage):
    @step
    async def maybe(self, upstream=optional_depends(ProduceStage.doc)) -> str:
        return "missing" if upstream is None else f"got {upstream.name}"

    steps = (maybe,)


def test_optional_dep_missing_reads_as_none(run):
    # ProduceStage never runs, so Produce/doc is absent; optional_depends -> None,
    # exercising InMemoryPersistService.read raising FileNotFoundError under the hood.
    svc = InMemoryPersistService()
    p = Pipeline(
        name="p",
        persist_service=svc,
        stages={"maybe": lambda ps: MaybeStage(persist_service=ps)},
    )
    assert run(p.run(module="maybe")) == ["missing"]
    assert svc.fetch("Maybe/maybe", str) == "missing"
