"""Local checkpoint management for learned denoisers."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn


class CheckpointManager:
    """Save the best model locally; checkpoints are never committed."""

    def __init__(self, directory: Path, *, monitor: str = "val_psnr", mode: str = "max") -> None:
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.directory = directory
        self.monitor = monitor
        self.mode = mode
        self.best_score: float | None = None
        self.best_epoch: int | None = None
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def best_path(self) -> Path:
        return self.directory / "best.pt"

    def is_improvement(self, score: float, min_delta: float = 0.0) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + min_delta
        return score < self.best_score - min_delta

    def save(
        self,
        state: dict[str, object],
        score: float,
        epoch: int,
        *,
        min_delta: float = 0.0,
    ) -> bool:
        if not self.is_improvement(score, min_delta):
            return False
        payload = {**state, "monitor": self.monitor, "score": score, "epoch": epoch}
        temporary = self.best_path.with_suffix(".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, self.best_path)
        self.best_score = score
        self.best_epoch = epoch
        return True

    @staticmethod
    def load(
        path: Path,
        model: nn.Module,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: object | None = None,
        device: str | torch.device = "cpu",
    ) -> dict[str, object]:
        payload = torch.load(path, map_location=device, weights_only=False)
        if "model_state" not in payload:
            raise ValueError(f"checkpoint lacks model_state: {path}")
        model.load_state_dict(payload["model_state"])
        if optimizer is not None and "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None and "scheduler_state" in payload:
            scheduler.load_state_dict(payload["scheduler_state"])
        return payload
