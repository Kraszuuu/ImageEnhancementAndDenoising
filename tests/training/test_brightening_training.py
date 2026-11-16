from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lowlight.models import build_model
from lowlight.training.brightening_datasets import (
    BrighteningDataset,
    BrighteningRow,
    d_fixed_cache_path,
    select_evenly,
)
from lowlight.training.brightening_trainer import BrighteningTrainer
from lowlight.training.checkpointing import CheckpointManager
from lowlight.training.losses import L1Loss
from lowlight.training.schedulers import build_scheduler


def _make_row(tmp_path: Path, name: str, value: float) -> BrighteningRow:
    low = tmp_path / f"{name}_low.png"
    gt = tmp_path / f"{name}_gt.png"
    array = np.full((32, 32, 3), value, dtype=np.uint8)
    Image.fromarray(array).save(low)
    Image.fromarray(np.full((32, 32, 3), 180, dtype=np.uint8)).save(gt)
    return BrighteningRow(
        dataset="lol",
        split="train",
        scene_id=f"scene_{name}",
        image_id=f"data/{name}.png",
        low_path=low,
        gt_path=gt,
    )


def _cache(tmp_path: Path, row: BrighteningRow, value: float) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    path = d_fixed_cache_path(cache_dir, row.image_id)
    np.save(path, np.full((32, 32, 3), value, dtype=np.float16))


def test_cache_path_is_stable(tmp_path: Path) -> None:
    assert d_fixed_cache_path(tmp_path, "a") == d_fixed_cache_path(tmp_path, "a")
    assert d_fixed_cache_path(tmp_path, "a") != d_fixed_cache_path(tmp_path, "b")


def test_select_evenly_per_dataset(tmp_path: Path) -> None:
    rows = [_make_row(tmp_path, f"a{index}", 0.2) for index in range(3)]
    rows += [
        BrighteningRow("lolv2-real", "train", f"s{index}", f"b{index}.png", Path("x"), Path("y"))
        for index in range(2)
    ]
    selected = select_evenly(rows, 2)
    assert len(selected) == 4
    assert {row.dataset for row in selected} == {"lol", "lolv2-real"}


def test_raw_dataset_returns_patch_and_target(tmp_path: Path) -> None:
    row = _make_row(tmp_path, "raw", 20)
    dataset = BrighteningDataset(
        [row], track="raw", training=True, seed=0, patch_size=16, augment=False
    )
    sample = dataset[0]
    assert sample["input"].shape == (3, 16, 16)
    assert sample["target"].shape == (3, 16, 16)
    assert sample["input"].max() <= 1.0
    assert sample["target"].min() >= 0.0


def test_d_fixed_dataset_uses_cache(tmp_path: Path) -> None:
    row = _make_row(tmp_path, "fixed", 20)
    _cache(tmp_path, row, 0.4)
    dataset = BrighteningDataset(
        [row],
        track="d_fixed",
        training=False,
        cache_dir=tmp_path / "cache",
    )
    sample = dataset[0]
    assert torch.allclose(sample["input"].mean(), torch.tensor(0.4), atol=2e-3)


def test_d_fixed_dataset_rejects_missing_cache(tmp_path: Path) -> None:
    row = _make_row(tmp_path, "missing", 20)
    dataset = BrighteningDataset(
        [row], track="d_fixed", training=False, cache_dir=tmp_path / "cache"
    )
    try:
        dataset[0]
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing cache should raise FileNotFoundError")


def test_brightening_trainer_one_epoch(tmp_path: Path) -> None:
    rows = [_make_row(tmp_path, f"train{index}", 20) for index in range(4)]
    dataset = BrighteningDataset(
        rows, track="raw", training=True, seed=0, patch_size=16, augment=False
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    val_dataset = BrighteningDataset(rows, track="raw", training=False)
    val_samples = [val_dataset[index] for index in range(len(val_dataset))]
    model = build_model(
        {
            "base_channels": 4,
            "depth": 2,
            "groups": 2,
            "residual": False,
        }
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_scheduler(optimizer, name="cosine", total_epochs=1, warmup_epochs=0)
    trainer = BrighteningTrainer(
        model=model,
        loss_fn=L1Loss(),
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        val_samples=val_samples,
        checkpoint=CheckpointManager(tmp_path / "checkpoints"),
        run_dir=tmp_path / "run",
        config={},
        seed=0,
        device="cpu",
        amp=False,
        max_epochs=1,
        patience=1,
    )
    result = trainer.fit()
    assert result["epochs_ran"] == 1
    assert np.isfinite(result["best_score"])
    assert (tmp_path / "run" / "history.csv").exists()
    assert (tmp_path / "checkpoints" / "best.pt").exists()
