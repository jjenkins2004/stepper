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

`DiskPersistService` stores each value under `base_dir` (plus a `run_id` subdir when given),
one file per key: a `str` as `<key>.txt`, raw `bytes` as `<key>` verbatim, everything else
as `<key>.json` (round-trips int/list/BaseModel). `base_dir` is required — the caller owns
where output lands.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar, cast

from pydantic import BaseModel, TypeAdapter

T = TypeVar("T")


def _is_persistable(model: object) -> bool:
    return isinstance(model, type) and issubclass(model, Persistable)


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
    def __init__(self, *, run_id: str | None = None) -> None:
        # The run this backend persists for; a subclass may key/partition output by it.
        self.run_id = run_id

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
    def __init__(self, base_dir: Path, *, run_id: str | None = None) -> None:
        """`base_dir` is the root dir output lands in; `run_id`, when given, is a
        per-run subdir under it (so `base_dir/run_id`)."""
        super().__init__(run_id=run_id)
        self._base = base_dir / run_id if run_id is not None else base_dir

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
