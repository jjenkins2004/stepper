"""Minimal, swappable persistence for pipeline steps.

`PersistService` is the interface (`persist` + `fetch`, plus an optional `run_id` it
carries so a backend can key/partition output by run) — swap in any backend.
`DiskPersistService` stores each step's value under `base_dir` (plus a `run_id` subdir
when given): a pure `str` as `<key>.txt`, anything else as `<key>.json` via a pydantic
`TypeAdapter` (which round-trips int/list/BaseModel/etc.). `base_dir` is required — the
caller owns where output lands.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar, cast

from pydantic import TypeAdapter

T = TypeVar("T")


class PersistService(ABC):
    def __init__(self, *, run_id: str | None = None) -> None:
        # The run this backend persists for; a subclass may key/partition output by it.
        self.run_id = run_id

    @abstractmethod
    def persist(self, key: str, value: T, model: type[T]) -> None:
        """Store one step's output.

        Args:
            key: Storage key, always "<stage_name>/<step_name>" (e.g. "Extract/build_order").
            value: The step's return value.
            model: Its type, used to serialize (`str` is stored as-is; anything else is
                dumped via a pydantic TypeAdapter).
        """
        ...

    @abstractmethod
    def fetch(self, key: str, model: type[T]) -> T:
        """Load back the value stored under `key` (the same key its producer used),
        decoded as `model`. Mirror of `persist`."""
        ...


class DiskPersistService(PersistService):
    def __init__(self, base_dir: Path, *, run_id: str | None = None) -> None:
        """`base_dir` is the root dir output lands in; `run_id`, when given, is a
        per-run subdir under it (so `base_dir/run_id`)."""
        super().__init__(run_id=run_id)
        self._base = base_dir / run_id if run_id is not None else base_dir

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
