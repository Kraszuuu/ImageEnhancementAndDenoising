from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from lowlight.training.brightening_datasets import BrighteningRow
from lowlight.training.pipeline_datasets import (
    EndToEndDataset,
    PipelineBrighteningDataset,
    source_cache_path,
    to_end_to_end_rows,
)
from lowlight.utils.noise import load_noise_config

ROOT = Path(__file__).resolve().parents[2]


def _make_row(tmp_path: Path, name: str) -> BrighteningRow:
    low_path = tmp_path / f"{name}_low.png"
    gt_path = tmp_path / f"{name}_gt.png"
    Image.fromarray(np.full((32, 32, 3), 30, dtype=np.uint8)).save(low_path)
    Image.fromarray(np.full((32, 32, 3), 180, dtype=np.uint8)).save(gt_path)
    return BrighteningRow(
        dataset="lol",
        split="train",
        scene_id=f"scene_{name}",
        image_id=f"data/{name}.png",
        low_path=low_path,
        gt_path=gt_path,
    )


def test_pipeline_brightening_dataset_uses_cache(tmp_path: Path) -> None:
    row = _make_row(tmp_path, "a")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    np.save(
        source_cache_path(cache_dir, row.image_id),
        np.full((32, 32, 3), 0.4, dtype=np.float16),
    )
    dataset = PipelineBrighteningDataset([row], cache_dir=cache_dir, training=False)
    sample = dataset[0]
    assert torch.allclose(sample["input"].mean(), torch.tensor(0.4), atol=2e-3)
    assert sample["target"].shape == (3, 32, 32)


def test_pipeline_brightening_dataset_rejects_missing_cache(tmp_path: Path) -> None:
    row = _make_row(tmp_path, "a")
    dataset = PipelineBrighteningDataset([row], cache_dir=tmp_path / "missing", training=False)
    with pytest.raises(FileNotFoundError, match="precomputePipelineSources"):
        dataset[0]


def test_end_to_end_real_uses_low_input(tmp_path: Path) -> None:
    row = _make_row(tmp_path, "a")
    dataset = EndToEndDataset(to_end_to_end_rows([row]), input_mode="real", training=False)
    sample = dataset[0]
    assert sample["level"] == "real"
    assert float(sample["input"].mean()) == pytest.approx(30 / 255, abs=1e-3)
    assert float(sample["target"].mean()) == pytest.approx(180 / 255, abs=1e-3)


def test_end_to_end_controlled_adds_noise(tmp_path: Path) -> None:
    row = _make_row(tmp_path, "a")
    noise_config = load_noise_config(ROOT / "configs" / "data" / "noise.yaml")
    dataset = EndToEndDataset(
        to_end_to_end_rows([row]),
        input_mode="controlled",
        noise_config=noise_config,
        training=False,
        level="high",
        noise_seed=0,
    )
    first = dataset[0]
    second = dataset[0]
    assert first["level"] == "high"
    assert torch.equal(first["input"], second["input"])
    assert float(first["input"].mean()) != pytest.approx(30 / 255, abs=1e-4)
    assert torch.equal(first["target"], second["target"])


def test_end_to_end_training_patch_shape(tmp_path: Path) -> None:
    row = _make_row(tmp_path, "a")
    noise_config = load_noise_config(ROOT / "configs" / "data" / "noise.yaml")
    dataset = EndToEndDataset(
        to_end_to_end_rows([row]),
        input_mode="controlled",
        noise_config=noise_config,
        training=True,
        seed=0,
        patch_size=16,
        augment=False,
    )
    sample = dataset[0]
    assert sample["input"].shape == (3, 16, 16)
    assert sample["target"].shape == (3, 16, 16)
