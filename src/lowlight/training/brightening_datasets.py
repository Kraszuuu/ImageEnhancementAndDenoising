"""Datasets for CNN brightening on paired low-light images."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from lowlight.utils.io import load_rgb, read_csv_rows

TRACKS = ("d_fixed", "raw")


@dataclass(frozen=True)
class BrighteningRow:
    dataset: str
    split: str
    scene_id: str
    image_id: str
    low_path: Path
    gt_path: Path


def load_brightening_rows(root: Path, split_paths: list[Path]) -> list[BrighteningRow]:
    rows = []
    for split_path in split_paths:
        for row in read_csv_rows(split_path):
            rows.append(
                BrighteningRow(
                    dataset=row["dataset"],
                    split=row["split"],
                    scene_id=row["scene_id"],
                    image_id=row["low_path"],
                    low_path=root / row["low_path"],
                    gt_path=root / row["gt_path"],
                )
            )
    return rows


def select_evenly(rows: list[BrighteningRow], per_dataset: int) -> list[BrighteningRow]:
    if per_dataset < 1:
        return []
    selected = []
    for dataset in sorted({row.dataset for row in rows}):
        subset = [row for row in rows if row.dataset == dataset]
        step = max(1, len(subset) // per_dataset)
        selected.extend(subset[::step][:per_dataset])
    return selected


def d_fixed_cache_path(cache_dir: Path, image_id: str) -> Path:
    digest = hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:24]
    return cache_dir / f"{digest}.npy"


class BrighteningDataset(Dataset):
    """Return ``input``/``target`` tensors in ``[0, 1]`` for one track."""

    def __init__(
        self,
        rows: list[BrighteningRow],
        *,
        track: str,
        training: bool,
        cache_dir: Path | None = None,
        seed: int = 0,
        patch_size: int | None = None,
        augment: bool = True,
        fixed_crop: bool = False,
    ) -> None:
        if track not in TRACKS:
            raise ValueError(f"unknown track: {track!r}")
        if training and patch_size is None:
            raise ValueError("training requires patch_size")
        if track == "d_fixed" and cache_dir is None:
            raise ValueError("d_fixed track requires cache_dir")
        self.rows = rows
        self.track = track
        self.training = training
        self.cache_dir = cache_dir
        self.seed = seed
        self.patch_size = patch_size
        self.augment = augment
        self.fixed_crop = fixed_crop
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def _rng(self, index: int) -> np.random.Generator:
        entropy = self.seed * 1_000_003 + self.epoch * 10_007 + index
        return np.random.default_rng(entropy)

    def _load_source(self, row: BrighteningRow) -> np.ndarray:
        if self.track == "d_fixed":
            if self.cache_dir is None:
                raise ValueError("d_fixed track requires cache_dir")
            path = d_fixed_cache_path(self.cache_dir, row.image_id)
            if not path.exists():
                raise FileNotFoundError(
                    f"missing D_fixed cache for {row.image_id}: {path}; "
                    "run scripts/precomputeDFixed.py"
                )
            return np.load(path).astype(np.float64)
        return load_rgb(row.low_path)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        source = self._load_source(row)
        target = load_rgb(row.gt_path)
        if source.shape != target.shape:
            raise ValueError(
                f"source/target shape mismatch for {row.image_id}: {source.shape} != {target.shape}"
            )
        rng = self._rng(index)
        if self.training:
            size = int(self.patch_size or 0)
            height, width = source.shape[:2]
            if height < size or width < size:
                raise ValueError(f"image too small for patch {size}: {row.image_id}")
            if self.fixed_crop:
                top = (height - size) // 2
                left = (width - size) // 2
            else:
                top = int(rng.integers(0, height - size + 1))
                left = int(rng.integers(0, width - size + 1))
            source = source[top : top + size, left : left + size]
            target = target[top : top + size, left : left + size]
            if self.augment:
                if rng.random() < 0.5:
                    source = source[:, ::-1]
                    target = target[:, ::-1]
                if rng.random() < 0.5:
                    source = source[::-1, :]
                    target = target[::-1, :]
                turns = int(rng.integers(0, 4))
                source = np.rot90(source, turns)
                target = np.rot90(target, turns)

        source = np.ascontiguousarray(source)
        target = np.ascontiguousarray(target)
        return {
            "dataset": row.dataset,
            "split": row.split,
            "image_id": row.image_id,
            "track": self.track,
            "input": torch.from_numpy(source.astype(np.float32)).permute(2, 0, 1),
            "target": torch.from_numpy(target.astype(np.float32)).permute(2, 0, 1),
        }
