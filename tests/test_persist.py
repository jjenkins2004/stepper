"""The persistence layer. The only contract a backend owes is that `write` then `read`
round-trips; everything above it treats keys as opaque strings."""

import pytest
from pydantic import BaseModel, PrivateAttr

from stepper import (
    EXIT,
    START,
    DiskPersistService,
    Flow,
    InMemoryPersistService,
    Persistable,
    PersistService,
    depends,
    edge,
    step,
)


class Doc(BaseModel):
    title: str
    pages: int


class Blob(Persistable):
    """A model with a side-artifact the plain encoding wouldn't cover."""

    caption: str
    _data: bytes = PrivateAttr(default=b"")
    _service: PersistService | None = PrivateAttr(default=None)
    _key: str = PrivateAttr(default="")

    def on_persist(self, service: PersistService, key: str) -> None:
        service.persist(f"{key}/blob.bin", self._data, bytes)

    def on_fetch(self, service: PersistService, key: str) -> None:
        self._service, self._key = service, key

    def load(self) -> bytes:
        assert self._service is not None
        return self._service.fetch(f"{self._key}/blob.bin", bytes)


@pytest.fixture(params=["disk", "memory"])
def backend(request, tmp_path):
    return DiskPersistService(base_dir=tmp_path) if request.param == "disk" else InMemoryPersistService()


# --- round trips ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,model",
    [
        (Doc(title="t", pages=3), Doc),
        ("plain text", str),
        (b"\x00\x01binary", bytes),
        (7, int),
        ([1, 2, 3], list[int]),
    ],
)
def test_values_round_trip(backend, value, model):
    backend.persist("k", value, model)
    assert backend.fetch("k", model) == value


def test_a_read_returns_an_independent_copy(backend):
    doc = Doc(title="t", pages=1)
    backend.persist("k", doc, Doc)
    fetched = backend.fetch("k", Doc)
    fetched.pages = 99
    assert backend.fetch("k", Doc).pages == 1


def test_a_missing_key_raises_file_not_found(backend):
    """Both backends agree, which is what lets an optional dep read back as None."""
    with pytest.raises(FileNotFoundError):
        backend.fetch("absent", Doc)


def test_a_key_with_slashes_nests(backend):
    backend.persist("a/b/c", Doc(title="t", pages=1), Doc)
    assert backend.fetch("a/b/c", Doc).title == "t"


# --- the disk encoding ----------------------------------------------------------------


def test_disk_picks_an_extension_from_the_model(tmp_path, persist):
    persist.persist("s", "text", str)
    persist.persist("d", Doc(title="t", pages=1), Doc)
    persist.persist("b", b"raw", bytes)
    assert (tmp_path / "s.txt").read_text() == "text"
    assert (tmp_path / "d.json").exists()
    assert (tmp_path / "b").read_bytes() == b"raw"


# --- Persistable ----------------------------------------------------------------------


def test_a_persistable_stores_its_side_artifact(backend):
    blob = Blob(caption="hi")
    blob._data = b"\xff\xfe"
    backend.persist("k", blob, Blob)
    back = backend.fetch("k", Blob)
    assert back.caption == "hi"
    assert back.load() == b"\xff\xfe"


def test_a_persistable_flows_through_a_run(run, mem):
    class Blobby(Flow[Blob]):
        @step
        async def make(self) -> Blob:
            b = Blob(caption="one")
            b._data = b"payload"
            return b

        @step
        async def read(self, b=depends(make)) -> Blob:
            out = Blob(caption=f"{b.caption}/{b.load().decode()}")
            out._data = b""
            return out

        edges = (edge(START).to(make), edge(make).to(read), edge(read).to(EXIT))

    assert run(Blobby().run(persist_service=mem)).caption == "one/payload"


# --- the backend owns where things land ------------------------------------------------


def test_the_backend_decides_the_location_not_the_flow(run, tmp_path):
    from _helpers import RootFlow

    nested = tmp_path / "deep" / "er"
    run(RootFlow().run(persist_service=DiskPersistService(base_dir=nested)))
    assert (nested / "root" / "start.json").exists()


def test_two_runs_of_one_flow_stay_apart_by_run_id(run, mem):
    from _helpers import Item, RootFlow

    run(RootFlow().run(run_id="a", persist_service=mem))
    run(RootFlow().run(run_id="b", persist_service=mem))
    assert mem.fetch("a/root/start", Item) == mem.fetch("b/root/start", Item)
    assert {k.split("/", 1)[0] for k in mem._store} == {"a", "b"}


# --- a Persistable buried in a plain model is rejected ---------------------------------
#
# The hooks only run for the *declared* model, so a Persistable used as a field of a plain
# BaseModel would have its side-artifacts dropped with nothing raised. That is caught where
# the step declares its return type.


class Bundle(BaseModel):
    """The mistake: a plain model carrying a Persistable."""

    note: str
    blob: Blob


class Outer(BaseModel):
    inner: Bundle


class SelfRef(BaseModel):
    """Guards the field walk against recursing forever."""

    name: str
    child: "SelfRef | None" = None


class Carrier(Persistable):
    """The supported way to nest: the outer model is itself a Persistable, so its own
    hooks decide what to forward."""

    blob: Blob

    def on_persist(self, service: PersistService, key: str) -> None:
        self.blob.on_persist(service, f"{key}/blob")

    def on_fetch(self, service: PersistService, key: str) -> None:
        self.blob.on_fetch(service, f"{key}/blob")


@pytest.mark.parametrize(
    "annotation,expected",
    [
        (Bundle, "Bundle.blob"),
        (Bundle | None, "Bundle.blob"),
        (list[Bundle], "Bundle.blob"),
        (dict[str, Bundle], "Bundle.blob"),
        (Outer, "Outer.inner -> Bundle.blob"),
    ],
)
def test_a_persistable_inside_a_plain_model_is_rejected(annotation, expected):
    with pytest.raises(TypeError) as exc:

        @step
        async def make(self) -> annotation: ...

    assert expected in str(exc.value)
    assert "Persistable" in str(exc.value)


@pytest.mark.parametrize(
    "annotation",
    [
        list[Blob],
        dict[str, Blob],
        Blob | None,
    ],
)
def test_a_persistable_inside_a_container_is_rejected(annotation):
    with pytest.raises(TypeError, match="Blob"):

        @step
        async def make(self) -> annotation: ...


@pytest.mark.parametrize(
    "annotation",
    [
        Blob,          # the declared model itself — hooks run
        Carrier,       # a Persistable holding one, forwarding by hand
        Doc,           # a plain model with nothing buried
        str,
        bytes,
        int,
        list[str],
        dict[str, int],
        SelfRef,       # recursive plain model, no Persistable anywhere
        None,
    ],
)
def test_models_without_a_buried_persistable_are_accepted(annotation):
    @step
    async def make(self) -> annotation: ...

    assert make.name == "make"


def test_the_error_names_the_field_and_the_fix():
    with pytest.raises(TypeError) as exc:

        @step
        async def build_bundle(self) -> Bundle: ...

    message = str(exc.value)
    assert "build_bundle" in message
    assert "Bundle.blob" in message
    assert "on_persist" in message


def test_a_carrier_persistable_still_round_trips_its_nested_blob(run, mem):
    """The supported nesting keeps working end to end."""

    class Nested(Flow[Carrier]):
        @step
        async def make(self) -> Carrier:
            inner = Blob(caption="c")
            inner._data = b"deep"
            return Carrier(blob=inner)

        edges = (edge(START).to(make), edge(make).to(EXIT))

    run(Nested().run(persist_service=mem))
    assert mem.fetch("nested/make", Carrier).blob.load() == b"deep"
