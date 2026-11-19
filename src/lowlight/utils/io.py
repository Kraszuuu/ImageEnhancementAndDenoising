"""Small I/O helpers shared by scripts and pipelines."""

from __future__ import annotations

import csv
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


def _handle_to_rgb(handle: Image.Image) -> np.ndarray:
    array = np.asarray(handle.convert("RGB"), dtype=np.float64) / 255.0
    return np.clip(array, 0.0, 1.0)


def load_rgb(path: Path) -> np.ndarray:
    """Load an image as float64 RGB in ``[0, 1]``."""
    with Image.open(path) as handle:
        return _handle_to_rgb(handle)


def load_rgb_bytes(data: bytes) -> np.ndarray:
    """Decode image bytes to float64 RGB exactly like :func:`load_rgb`."""
    with Image.open(BytesIO(data)) as handle:
        return _handle_to_rgb(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(
    path: Path, rows: list[dict[str, object]], columns: list[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
