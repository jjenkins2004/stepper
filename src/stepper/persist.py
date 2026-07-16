"""Minimal, swappable persistence for pipeline steps.

`PersistService` is the interface (`persist` + `fetch`, plus an optional `run_id` it
carries so a backend can key/partition output by run) — swap in any backend.
`DiskPersistService` stores each step's value under `base_dir` (plus a `run_id` subdir
when given): a pure `str` as `<key>.txt`, a `PIL.Image.Image` as a viewable image file
keeping its own format (`<key>.<fmt>`, e.g. `.jpg`/`.png`; PNG when the image has no
format, as created/processed ones don't), anything else as `<key>.json` via a pydantic
`TypeAdapter` (which round-trips int/list/BaseModel/etc.). `base_dir` is required — the
caller owns where output lands.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar, cast

from PIL import Image
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

    @staticmethod
    def _is_image(model: type) -> bool:
        return isinstance(model, type) and issubclass(model, Image.Image)

    def persist(self, key: str, value: T, model: type[T]) -> None:
        if self._is_image(model):
            img = cast(Image.Image, value)
            fmt = img.format or "PNG"  # created/processed images carry no format
            ext = "jpg" if fmt == "JPEG" else fmt.lower()
            path = self._base / f"{key}.{ext}"
            path.parent.mkdir(parents=True, exist_ok=True)
            img.save(path, format=fmt)
            return
        path = self._base / f"{key}.{'txt' if model is str else 'json'}"
        path.parent.mkdir(parents=True, exist_ok=True)
        if model is str:
            path.write_text(cast(str, value), encoding="utf-8")
        else:
            path.write_bytes(TypeAdapter(model).dump_json(value, indent=2))

    def fetch(self, key: str, model: type[T]) -> T:
        if self._is_image(model):
            # persist keeps each image's own format, so the extension isn't known
            # from the type — find the one file for this key; PIL reads the format
            # from its bytes regardless of extension.
            matches = list(self._base.glob(f"{key}.*"))
            if not matches:
                raise FileNotFoundError(f"no image stored for {key!r} under {self._base}")
            with Image.open(matches[0]) as im:
                im.load()  # read pixels now so the file handle can close
                return cast(T, im)
        if model is str:
            return cast(T, (self._base / f"{key}.txt").read_text(encoding="utf-8"))
        return TypeAdapter(model).validate_json((self._base / f"{key}.json").read_bytes())
