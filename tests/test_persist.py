"""Persistence: str/bytes/JSON values + the Persistable side-artifact hooks.

No ``from __future__ import annotations`` — ``Step`` reads raw ``fn.__annotations__``,
so stringized return annotations would break ``.model`` inference for the stages below.

Uses the disk-backed ``persist`` fixture (see conftest); ``tmp_path`` is the same dir it
roots under, so tests can assert the on-disk path directly.
"""

import pytest
from pydantic import PrivateAttr

from stepper import (
    DiskPersistService,
    Persistable,
    PersistService,
    Pipeline,
    Stage,
    depends,
    step,
)


class Doc(Persistable):
    """Toy Persistable: JSON metadata (``name`` + the blob ``ids``) plus a PrivateAttr
    blob map persisted as ``bytes`` under sub-keys and lazy-read after fetch via ``load``."""

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


# --- raw bytes under an opaque key -------------------------------------------------


def test_bytes_persist_under_opaque_key(persist, tmp_path):
    # bytes land under the key verbatim — any extension is just part of the key
    persist.persist("Render/thumb.png", b"\x00\x01\x02", bytes)

    assert (tmp_path / "Render" / "thumb.png").exists()
    assert persist.fetch("Render/thumb.png", bytes) == b"\x00\x01\x02"


def test_fetch_missing_bytes_raises(persist):
    with pytest.raises(FileNotFoundError, match="Render/thumb"):
        persist.fetch("Render/thumb.png", bytes)


# --- Persistable metadata + blobs --------------------------------------------------


def test_persistable_round_trips_metadata_and_blobs(persist, tmp_path):
    doc = Doc(name="report", ids=["a", "b"])
    doc._blobs = {"a": b"alpha", "b": b"beta"}
    persist.persist("Build/doc", doc, Doc)

    # metadata JSON and the blob dir coexist under the same key
    assert (tmp_path / "Build" / "doc.json").exists()
    assert (tmp_path / "Build" / "doc" / "a.bin").exists()
    assert (tmp_path / "Build" / "doc" / "b.bin").exists()

    back = persist.fetch("Build/doc", Doc)
    assert back.name == "report"
    assert back.ids == ["a", "b"]
    assert not back._blobs  # blobs aren't in the JSON; loaded lazily
    assert back.load("a") == b"alpha"
    assert back.load("b") == b"beta"


# --- through a two-stage Pipeline with run_id --------------------------------------


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


def test_persistable_flows_through_pipeline(tmp_path, run):
    p = Pipeline(
        name="p",
        run_id="r1",
        output_root=tmp_path,
        stages={
            "produce": lambda ps: ProduceStage(persist_service=ps),
            "consume": lambda ps: ConsumeStage(persist_service=ps),
        },
    )
    run(p.run(module="all"))

    base = tmp_path / "p" / "r1"  # output_root/name/run_id
    assert (base / "Produce" / "doc.json").exists()
    assert (base / "Produce" / "doc" / "a.bin").exists()
    # consumer depends() on the Persistable, fetched back and its blob lazy-loaded
    assert p.persist_service.fetch("Consume/read", str) == "report:payload"


def test_run_id_scopes_persistable_output(tmp_path):
    svc = DiskPersistService(base_dir=tmp_path, run_id="r9")
    doc = Doc(name="x", ids=["a"])
    doc._blobs = {"a": b"z"}
    svc.persist("Build/doc", doc, Doc)

    assert (tmp_path / "r9" / "Build" / "doc.json").exists()
    assert (tmp_path / "r9" / "Build" / "doc" / "a.bin").exists()
