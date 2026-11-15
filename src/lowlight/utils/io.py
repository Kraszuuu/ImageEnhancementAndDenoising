"""Small I/O helpers shared by scripts and pipelines."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path: Path) -> np.ndarray:
    """Load an image as float64 RGB in ``[0, 1]``."""
    with Image.open(path) as handle:
        array = np.asarray(handle.convert("RGB"), dtype=np.float64) / 255.0
    return np.clip(array, 0.0, 1.0)


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
