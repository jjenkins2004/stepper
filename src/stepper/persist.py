"""Minimal, swappable persistence for pipeline steps.

`PersistService` is the interface — swap in any backend. `persist`/`fetch` are the entry
points: they hand a step's value to the one abstract method pair a backend implements
(`write`/`read`) to store and reload it, then (for a `Persistable`) run the model's own
`on_persist`/`on_fetch` hooks for any side-artifacts. How a value is encoded and where it
lands is entirely the backend's business — the only contract is that `write` then `read`
round-trips.

The core knows nothing about images or any specific binary format. A consumer that needs
more than the backend's plain encoding subclasses `Persistable` and does its own
persistence in the hooks, baking any backend naming (a file extension, a bucket path) into
the keys it passes.

The hooks run for the model `persist`/`fetch` are *given*, not for models reached through
its fields, so a `Persistable` buried in a plain `BaseModel` would silently lose its
side-artifacts. `nested_persistable_path` finds that, and `Step` rejects it where the return
type is declared.

`DiskPersistService` stores each value under `base_dir`, one file per key: a `str` as
`<key>.txt`, raw `bytes` as `<key>` verbatim, everything else as `<key>.json` (round-trips
int/list/BaseModel). `base_dir` is required — a backend owns where output physically
lands, which is why it, and not a flow, is what you configure. Per-run separation is the
flow's `run_id`, which arrives baked into the keys.

`InMemoryPersistService` is the same encoding into a plain dict instead of files — nothing
touches disk. Use it for a single server-side run whose output must not be written anywhere;
the store is discarded with the instance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeVar, cast, get_args, get_origin

from pydantic import BaseModel, TypeAdapter

T = TypeVar("T")


def _is_persistable(model: object) -> bool:
    return isinstance(model, type) and issubclass(model, Persistable)


def _is_plain_model(model: object) -> bool:
    return (
        isinstance(model, type)
        and issubclass(model, BaseModel)
        and not issubclass(model, Persistable)
    )


def nested_persistable_path(model: object) -> str | None:
    """The path to a `Persistable` buried inside a plain model, or `None` if there is none.

    `persist`/`fetch` decide whether to run the hooks with one check on the *declared*
    model — they don't walk fields. So a `Persistable` used as a field of a plain
    `BaseModel` has its fields dumped to JSON and its side-artifacts silently dropped:
    the metadata reads back intact and the blobs are simply gone. Catching that where the
    type is declared is the difference between an error and missing bytes.

    Descends through plain models, unions and containers. Stops at a `Persistable` — one
    nested inside another is that model's own business, since its `on_persist` decides
    what to forward.
    """
    return _search(model, frozenset())


def _search(model: object, seen: frozenset[Any]) -> str | None:
    if get_origin(model) is not None:
        # A union or container (`list[X]`, `dict[str, X]`, `X | None`) — a Persistable is
        # just as buried inside one, so look at what it holds.
        for arg in get_args(model):
            if arg is type(None):
                continue
            if _is_persistable(arg):
                return getattr(arg, "__name__", str(arg))
            found = _search(arg, seen)
            if found is not None:
                return found
        return None

    if not _is_plain_model(model) or model in seen:
        return None

    owner = cast(type, model).__name__
    seen = seen | {model}
    for name, field in cast(type[BaseModel], model).model_fields.items():
        annotation = field.annotation
        if _is_persistable(annotation):
            return f"{owner}.{name}"
        found = _search(annotation, seen)
        if found is not None:
            # A container hit reports just the type; a nested model hit reports a path.
            joiner = " -> " if "." in found else ": "
            return f"{owner}.{name}{joiner}{found}"
    return None


class Persistable(BaseModel, ABC):
    """A model that persists side-artifacts of its own alongside the usual field data.

    By default the service just serializes a model's fields. A `Persistable` adds two
    hooks around that: after its fields are written, `on_persist` runs so the model can
    store extra data the fields don't cover (a large blob, a file, something fetched
    elsewhere) by calling `persist` again under its own key; after it's loaded back,
    `on_fetch` runs so it can read that data — or keep `service`+`key` to load it lazily.
    Store the side data off the normal fields so it isn't serialized twice.

    Example — a model with a caption (a normal field) plus a large image blob::

        def on_persist(self, service, key):
            service.persist(f"{key}/image.png", self.image_bytes, bytes)

        def on_fetch(self, service, key):
            self.image_bytes = service.fetch(f"{key}/image.png", bytes)
    """

    @abstractmethod
    def on_persist(self, service: PersistService, key: str) -> None:
        """Runs after the model's fields are serialized. Persist any side-artifacts here,
        under a key of your own (e.g. `service.persist(f'{key}/image.png', data, bytes)`)."""

    @abstractmethod
    def on_fetch(self, service: PersistService, key: str) -> None:
        """Runs after the model is loaded back. Read your side-artifacts here, or hold onto
        `service` and `key` to load them lazily on first access."""


class PersistService(ABC):
    def persist(self, key: str, value: T, model: type[T]) -> None:
        """Store one step's output under `key` ("<stage_name>/<step_name>"): write the
        value, then (for a `Persistable`) run its `on_persist` for any side-artifacts."""
        self.write(key, value, model)
        if _is_persistable(model):
            cast(Persistable, value).on_persist(self, key)

    def fetch(self, key: str, model: type[T]) -> T:
        """Load back the value stored under `key`, decoded as `model`. Mirror of `persist`:
        read the value, then (for a `Persistable`) run its `on_fetch` to attach the service.
        Raises if nothing is stored under `key` — the caller decides whether that's fatal."""
        value = self.read(key, model)
        if _is_persistable(model):
            cast(Persistable, value).on_fetch(self, key)
        return value

    # Encode one value under `key` and read it back.
    @abstractmethod
    def write(self, key: str, value: T, model: type[T]) -> None: ...
    @abstractmethod
    def read(self, key: str, model: type[T]) -> T: ...


class DiskPersistService(PersistService):
    def __init__(self, base_dir: Path | str) -> None:
        """`base_dir` is the root dir output lands in. Nothing else: a run's separation
        comes from the `run_id` its root flow bakes into every key."""
        self._base = Path(base_dir)

    def write(self, key: str, value: T, model: type[T]) -> None:
        # str -> .txt, bytes -> the key verbatim, everything else -> .json
        ext = "" if model is bytes else f".{'txt' if model is str else 'json'}"
        path = self._base / f"{key}{ext}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if model is str:
            path.write_text(cast(str, value), encoding="utf-8")
        elif model is bytes:
            path.write_bytes(cast(bytes, value))
        else:
            path.write_bytes(TypeAdapter(model).dump_json(value, indent=2))

    def read(self, key: str, model: type[T]) -> T:
        if model is str:
            return cast(T, (self._base / f"{key}.txt").read_text(encoding="utf-8"))
        if model is bytes:
            return cast(T, (self._base / key).read_bytes())
        return TypeAdapter(model).validate_json((self._base / f"{key}.json").read_bytes())


class InMemoryPersistService(PersistService):
    """`PersistService` backed by a plain dict — nothing touches disk or any external store.

    For a single server-side run whose output must not be written anywhere: the store lives
    only as long as the instance. Encodes exactly like `DiskPersistService` so the two are
    interchangeable — a value is serialized on `write` (`str` verbatim, `bytes` verbatim,
    everything else to JSON) and decoded on `read`. Serializing (not stashing the live
    object) mirrors disk: a read returns an independent copy, so one step can't mutate an
    earlier step's persisted value.
    """

    def __init__(self) -> None:
        self._store: dict[str, str | bytes] = {}

    def write(self, key: str, value: T, model: type[T]) -> None:
        if model is str:
            self._store[key] = cast(str, value)
        elif model is bytes:
            self._store[key] = cast(bytes, value)
        else:
            self._store[key] = TypeAdapter(model).dump_json(value)

    def read(self, key: str, model: type[T]) -> T:
        try:
            raw = self._store[key]
        except KeyError:
            # Mirror the disk backend: a missing value raises FileNotFoundError, so an
            # optional dep reads back as None instead of raising (see the Stage runner).
            raise FileNotFoundError(key) from None
        if model is str or model is bytes:
            return cast(T, raw)
        return TypeAdapter(model).validate_json(cast(bytes, raw))
