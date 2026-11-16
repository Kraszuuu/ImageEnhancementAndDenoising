from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from lowlight.models import CompactUNet, UNetConfig
from lowlight.training.checkpointing import CheckpointManager
from lowlight.training.datasets import DenoisingDataset, SplitRow
from lowlight.training.denoising_trainer import DenoisingTrainer
from lowlight.training.losses import L1Loss, L1PerceptualLoss, build_loss
from lowlight.training.schedulers import WarmupCosine, build_scheduler
from lowlight.utils.noise import NoiseConfig, NoiseLevel


def _noise_config() -> NoiseConfig:
    return NoiseConfig(
        color_space="sRGB",
        range=(0.0, 1.0),
        apply_in="linear_light",
        seed_policy="fixed",
        levels={
            "clean": NoiseLevel(0.0, 0.0),
            "low": NoiseLevel(0.001, 0.002),
            "medium": NoiseLevel(0.01, 0.005),
            "high": NoiseLevel(0.05, 0.02),
        },
    )


def _write_image(path: Path, seed: int) -> None:
    image = (np.random.default_rng(seed).random((40, 48, 3)) * 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)


def test_dataset_training_shape_and_determinism(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    _write_image(image_path, 0)
    row = SplitRow("d", "train", str(image_path), image_path)
    dataset = DenoisingDataset(
        [row], _noise_config(), root=tmp_path, training=True, seed=0, patch_size=32
    )
    first = dataset[0]
    second = dataset[0]
    assert first["noisy"].shape == (3, 32, 32)
    assert torch.equal(first["noisy"], second["noisy"])
    assert torch.equal(first["clean"], second["clean"])
    dataset.set_epoch(1)
    assert not torch.equal(first["noisy"], dataset[0]["noisy"])


def test_dataset_validation_requires_level(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    _write_image(image_path, 1)
    row = SplitRow("d", "val", str(image_path), image_path)
    with pytest.raises(ValueError):
        DenoisingDataset([row], _noise_config(), root=tmp_path, training=False)
    dataset = DenoisingDataset(
        [row], _noise_config(), root=tmp_path, training=False, level="high"
    )
    sample = dataset[0]
    assert sample["level"] == "high"
    assert sample["noisy"].shape == (3, 40, 48)


def test_losses_and_build() -> None:
    prediction = torch.zeros(2, 3, 8, 8)
    target = torch.ones(2, 3, 8, 8)
    assert build_loss("l1")(prediction, target).item() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        build_loss("nope")
    with pytest.raises(ValueError):
        L1PerceptualLoss(lambda_lpips=-1.0)


def test_scheduler_warmup_and_decay() -> None:
    parameter = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = build_scheduler(
        optimizer, name="cosine", total_epochs=10, warmup_epochs=2, eta_min_factor=0.01
    )
    learning_rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(10):
        scheduler.step()
        learning_rates.append(optimizer.param_groups[0]["lr"])
    assert learning_rates[0] < 1.0
    assert learning_rates[2] <= 1.0
    assert learning_rates[-1] < 0.02
    assert all(rate > 0.0 for rate in learning_rates)
    with pytest.raises(ValueError):
        WarmupCosine(optimizer, total_epochs=4, warmup_epochs=4)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = nn.Linear(2, 1)
    manager = CheckpointManager(tmp_path)
    assert manager.save({"model_state": model.state_dict()}, 1.0, 1)
    assert not manager.save({"model_state": model.state_dict()}, 0.5, 2)
    loaded = nn.Linear(2, 1)
    payload = CheckpointManager.load(manager.best_path, loaded)
    assert payload["score"] == 1.0
    assert torch.equal(model.weight, loaded.weight)


class _TinyDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.items = [
            {
                "noisy": torch.rand(3, 8, 8),
                "clean": torch.rand(3, 8, 8),
                "dataset": "d",
                "level": "high",
            }
            for _ in range(2)
        ]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]


def test_trainer_runs_one_epoch_cpu(tmp_path: Path) -> None:
    model = CompactUNet(UNetConfig(base_channels=4, depth=2, groups=4))
    loader = DataLoader(_TinyDataset(), batch_size=2)
    val_samples = [
        {
            "noisy": torch.rand(3, 8, 8),
            "clean": torch.rand(3, 8, 8),
            "dataset": "d",
            "level": "high",
        }
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_scheduler(optimizer, name="cosine", total_epochs=1)
    trainer = DenoisingTrainer(
        model=model,
        loss_fn=L1Loss(),
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=loader,
        val_samples=val_samples,
        checkpoint=CheckpointManager(tmp_path / "ckpt"),
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
    assert (tmp_path / "ckpt" / "best.pt").exists()
    assert (tmp_path / "run" / "history.csv").exists()
