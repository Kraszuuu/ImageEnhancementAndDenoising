"""Datasets for Stage 06 pipeline-aware training."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from lowlight.training.brightening_datasets import BrighteningRow
from lowlight.utils.io import load_rgb
from lowlight.utils.noise import LEVEL_NAMES, NoiseConfig


def source_cache_path(cache_dir: Path, image_id: str) -> Path:
    digest = hashlib.sha256(image_id.encode("utf-8")).hexdigest()[:24]
    return cache_dir / f"{digest}.npy"


@dataclass(frozen=True)
class EndToEndRow:
    dataset: str
    split: str
    image_id: str
    low_path: Path
    gt_path: Path


def to_end_to_end_rows(rows: list[BrighteningRow]) -> list[EndToEndRow]:
    return [
        EndToEndRow(
            dataset=row.dataset,
            split=row.split,
            image_id=row.image_id,
            low_path=row.low_path,
            gt_path=row.gt_path,
        )
        for row in rows
    ]


class PipelineBrighteningDataset(Dataset):
    """Brightener dataset whose input is a cached denoiser output."""

    def __init__(
        self,
        rows: list[BrighteningRow],
        *,
        cache_dir: Path,
        training: bool,
        seed: int = 0,
        patch_size: int | None = None,
        augment: bool = True,
        fixed_crop: bool = False,
    ) -> None:
        if training and patch_size is None:
            raise ValueError("training requires patch_size")
        self.rows = rows
        self.cache_dir = cache_dir
        self.training = training
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
        path = source_cache_path(self.cache_dir, row.image_id)
        if not path.exists():
            raise FileNotFoundError(
                f"missing pipeline source cache for {row.image_id}: {path}; "
                "run scripts/precomputePipelineSources.py"
            )
        return np.load(path).astype(np.float64)

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
            "input": torch.from_numpy(source.astype(np.float32)).permute(2, 0, 1),
            "target": torch.from_numpy(target.astype(np.float32)).permute(2, 0, 1),
        }


class EndToEndDataset(Dataset):
    """Map real or controlled-noisy low-light input directly to normal light."""

    def __init__(
        self,
        rows: list[EndToEndRow],
        *,
        input_mode: str,
        noise_config: NoiseConfig | None = None,
        levels: tuple[str, ...] = ("low", "medium", "high"),
        training: bool,
        level: str | None = None,
        seed: int = 0,
        noise_seed: int = 0,
        patch_size: int | None = None,
        augment: bool = True,
        fixed_crop: bool = False,
    ) -> None:
        if input_mode not in {"real", "controlled"}:
            raise ValueError(f"unknown input_mode: {input_mode!r}")
        if input_mode == "controlled" and noise_config is None:
            raise ValueError("controlled input requires noise_config")
        if level is not None and level not in LEVEL_NAMES:
            raise ValueError(f"unknown noise level: {level!r}")
        if training and patch_size is None:
            raise ValueError("training requires patch_size")
        if not training and input_mode == "controlled" and level is None:
            raise ValueError("controlled evaluation requires a fixed level")
        self.rows = rows
        self.input_mode = input_mode
        self.noise_config = noise_config
        self.levels = tuple(levels)
        self.training = training
        self.level = level
        self.seed = seed
        self.noise_seed = noise_seed
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

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        low = load_rgb(row.low_path)
        target = load_rgb(row.gt_path)
        rng = self._rng(index)
        if self.training:
            size = int(self.patch_size or 0)
            height, width = low.shape[:2]
            if height < size or width < size:
                raise ValueError(f"image too small for patch {size}: {row.image_id}")
            if self.fixed_crop:
                top = (height - size) // 2
                left = (width - size) // 2
            else:
                top = int(rng.integers(0, height - size + 1))
                left = int(rng.integers(0, width - size + 1))
            low = low[top : top + size, left : left + size]
            target = target[top : top + size, left : left + size]
            if self.augment:
                if rng.random() < 0.5:
                    low = low[:, ::-1]
                    target = target[:, ::-1]
                if rng.random() < 0.5:
                    low = low[::-1, :]
                    target = target[::-1, :]
                turns = int(rng.integers(0, 4))
                low = np.rot90(low, turns)
                target = np.rot90(target, turns)

        if self.input_mode == "controlled":
            level = (
                self.levels[int(rng.integers(0, len(self.levels)))]
                if self.training
                else str(self.level)
            )
            assert self.noise_config is not None
            source = self.noise_config.add(
                np.ascontiguousarray(low),
                level,
                image_id=row.image_id,
                seed=self.noise_seed,
            )
        else:
            level = "real"
            source = np.ascontiguousarray(low)

        target = np.ascontiguousarray(target)
        return {
            "dataset": row.dataset,
            "split": row.split,
            "image_id": row.image_id,
            "level": level,
            "input": torch.from_numpy(source.astype(np.float32)).permute(2, 0, 1),
            "target": torch.from_numpy(target.astype(np.float32)).permute(2, 0, 1),
        }
