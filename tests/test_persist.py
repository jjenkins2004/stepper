"""DiskPersistService image logic — the PIL.Image.Image branch of persist/fetch.

Uses the disk-backed ``persist`` fixture (see conftest); ``tmp_path`` is the same dir it
roots under, so tests assert the on-disk path directly.
"""

import pytest
from PIL import Image


def test_created_image_round_trips_as_png(persist, tmp_path):
    """An image made in memory has no format, so it lands as a viewable .png and
    round-trips pixel-for-pixel (PNG is lossless)."""
    img = Image.new("RGB", (4, 3), (10, 20, 30))
    persist.persist("Render/thumb", img, Image.Image)

    assert (tmp_path / "Render" / "thumb.png").exists()
    back = persist.fetch("Render/thumb", Image.Image)
    assert back.format == "PNG"
    assert back.size == (4, 3)
    assert back.mode == "RGB"
    assert back.getpixel((0, 0)) == (10, 20, 30)


def test_rgba_alpha_survives_png_round_trip(persist):
    img = Image.new("RGBA", (2, 2), (5, 6, 7, 128))
    persist.persist("Render/mask", img, Image.Image)

    back = persist.fetch("Render/mask", Image.Image)
    assert back.mode == "RGBA"
    assert back.getpixel((0, 0)) == (5, 6, 7, 128)


def test_source_format_is_preserved(persist, tmp_path):
    """An image opened from a JPEG keeps format=='JPEG', so it's stored as .jpg
    (not .jpeg) and fetched back as JPEG."""
    src = tmp_path / "src.jpg"
    Image.new("RGB", (8, 8), (0, 120, 255)).save(src, format="JPEG")
    opened = Image.open(src)
    opened.load()
    assert opened.format == "JPEG"

    persist.persist("Load/photo", opened, Image.Image)
    assert (tmp_path / "Load" / "photo.jpg").exists()

    back = persist.fetch("Load/photo", Image.Image)
    assert back.format == "JPEG"
    assert back.size == (8, 8)


def test_fetch_closes_file_handle(persist):
    """fetch calls im.load() then exits the context manager, so the returned image is
    fully read and its file handle is released."""
    persist.persist("Render/thumb", Image.new("RGB", (2, 2)), Image.Image)

    back = persist.fetch("Render/thumb", Image.Image)
    assert back.fp is None  # handle closed, pixels already loaded
    assert back.getpixel((0, 0)) == (0, 0, 0)  # still usable after close


def test_fetch_missing_image_raises(persist):
    with pytest.raises(FileNotFoundError, match="Render/thumb"):
        persist.fetch("Render/thumb", Image.Image)
