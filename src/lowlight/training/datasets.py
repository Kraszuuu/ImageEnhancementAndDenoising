"""Datasets for on-the-fly Gaussian-Poisson denoising experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from lowlight.utils.io import load_rgb, read_csv_rows
from lowlight.utils.noise import LEVEL_NAMES, NoiseConfig


@dataclass(frozen=True)
class SplitRow:
    dataset: str
    split: str
    image_id: str
    path: Path


def load_split_rows(root: Path, split_paths: list[Path]) -> list[SplitRow]:
    rows = []
    for split_path in split_paths:
        for row in read_csv_rows(split_path):
            rows.append(
                SplitRow(
                    dataset=row["dataset"],
                    split=row["split"],
                    image_id=row["low_path"],
                    path=root / row["low_path"],
                )
            )
    return rows


def select_evenly(rows: list[SplitRow], per_dataset: int) -> list[SplitRow]:
    if per_dataset < 1:
        return []
    selected = []
    for dataset in sorted({row.dataset for row in rows}):
        subset = [row for row in rows if row.dataset == dataset]
        step = max(1, len(subset) // per_dataset)
        selected.extend(subset[::step][:per_dataset])
    return selected


class DenoisingDataset(Dataset):
    """Return clean/noisy tensors in ``[0, 1]`` with deterministic noise."""

    def __init__(
        self,
        rows: list[SplitRow],
        noise_config: NoiseConfig,
        *,
        root: Path,
        training: bool,
        seed: int = 0,
        noise_seed: int = 0,
        patch_size: int | None = None,
        level: str | None = None,
        levels: tuple[str, ...] = LEVEL_NAMES,
        augment: bool = True,
        fixed_crop: bool = False,
    ) -> None:
        if level is not None and level not in LEVEL_NAMES:
            raise ValueError(f"unknown noise level: {level!r}")
        if training and patch_size is None:
            raise ValueError("training requires patch_size")
        if not training and level is None:
            raise ValueError("evaluation requires a fixed level")
        self.rows = rows
        self.noise_config = noise_config
        self.root = root
        self.training = training
        self.seed = seed
        self.noise_seed = noise_seed
        self.patch_size = patch_size
        self.level = level
        self.levels = tuple(levels)
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

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        image = load_rgb(row.path)
        rng = self._rng(index)
        if self.training:
            level = self.levels[int(rng.integers(0, len(self.levels)))]
            size = int(self.patch_size or 0)
            height, width = image.shape[:2]
            if height < size or width < size:
                raise ValueError(f"image too small for patch {size}: {row.image_id}")
            if self.fixed_crop:
                top = (height - size) // 2
                left = (width - size) // 2
            else:
                top = int(rng.integers(0, height - size + 1))
                left = int(rng.integers(0, width - size + 1))
            image = image[top : top + size, left : left + size]
            if self.augment:
                if rng.random() < 0.5:
                    image = image[:, ::-1]
                if rng.random() < 0.5:
                    image = image[::-1, :]
                image = np.ascontiguousarray(np.rot90(image, int(rng.integers(0, 4))))
        else:
            level = str(self.level)
        clean = np.ascontiguousarray(image)
        noisy = self.noise_config.add(
            clean, level, image_id=row.image_id, seed=self.noise_seed
        )
        return {
            "dataset": row.dataset,
            "split": row.split,
            "image_id": row.image_id,
            "level": level,
            "clean": torch.from_numpy(clean.astype(np.float32)).permute(2, 0, 1),
            "noisy": torch.from_numpy(noisy.astype(np.float32)).permute(2, 0, 1),
        }
