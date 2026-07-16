"""Minimal, swappable persistence for pipeline steps.

`PersistService` is the interface (just `persist` + `fetch`) — swap in any backend.
`DiskPersistService` stores each step's value under `base_dir`: a pure `str` as
`<key>.txt`, anything else as `<key>.json` via a pydantic `TypeAdapter` (which
round-trips int/list/BaseModel/etc.). `base_dir` is required — the caller owns where
output lands.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar, cast

from pydantic import TypeAdapter

T = TypeVar("T")


class PersistService(ABC):
    @abstractmethod
    def persist(self, key: str, value: T, model: type[T]) -> None: ...

    @abstractmethod
    def fetch(self, key: str, model: type[T]) -> T: ...


class DiskPersistService(PersistService):
    def __init__(self, base_dir: Path) -> None:
        self._base = base_dir

    def persist(self, key: str, value: T, model: type[T]) -> None:
        path = self._base / f"{key}.{'txt' if model is str else 'json'}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if model is str:
            path.write_text(cast(str, value), encoding="utf-8")
        else:
            path.write_bytes(TypeAdapter(model).dump_json(value, indent=2))

    def fetch(self, key: str, model: type[T]) -> T:
        if model is str:
            return cast(T, (self._base / f"{key}.txt").read_text(encoding="utf-8"))
        return TypeAdapter(model).validate_json((self._base / f"{key}.json").read_bytes())
