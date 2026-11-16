"""Training and validation loop for CNN denoisers."""

from __future__ import annotations

import csv
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from lowlight.models import denoise_full_image
from lowlight.training.checkpointing import CheckpointManager


def _psnr(prediction: torch.Tensor, target: torch.Tensor) -> list[float]:
    mse = ((prediction - target) ** 2).mean(dim=(1, 2, 3))
    values = []
    for item in mse:
        if item.item() == 0.0:
            values.append(math.inf)
        else:
            values.append(float(10.0 * torch.log10(1.0 / item)))
    return values


class DenoisingTrainer:
    def __init__(
        self,
        *,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        train_loader: DataLoader,
        val_samples: list[dict[str, object]],
        checkpoint: CheckpointManager,
        run_dir: Path,
        config: dict[str, object],
        seed: int,
        device: str | torch.device = "cuda",
        amp: bool = True,
        grad_clip: float | None = 1.0,
        grad_accum_steps: int = 1,
        max_epochs: int = 30,
        patience: int = 8,
        min_delta: float = 1e-4,
        val_batch_size: int = 4,
    ) -> None:
        if max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        if patience < 1:
            raise ValueError("patience must be positive")
        if grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be positive")
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_samples = val_samples
        self.checkpoint = checkpoint
        self.run_dir = run_dir
        self.config = config
        self.seed = seed
        self.device = torch.device(device)
        self.amp = bool(amp and self.device.type == "cuda")
        self.grad_clip = grad_clip
        self.grad_accum_steps = grad_accum_steps
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.val_batch_size = val_batch_size
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.history_path = run_dir / "history.csv"
        self.history: list[dict[str, object]] = []

    def _train_epoch(self, epoch: int) -> float:
        dataset = self.train_loader.dataset
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)
        self.model.train()
        total_loss = 0.0
        total_items = 0
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(self.train_loader, start=1):
            noisy = batch["noisy"].to(self.device, non_blocking=True)
            clean = batch["clean"].to(self.device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                prediction = self.model(noisy)
                loss = self.loss_fn(prediction, clean)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at epoch {epoch} step {step}")
            self.scaler.scale(loss / self.grad_accum_steps).backward()
            if step % self.grad_accum_steps == 0:
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach()) * noisy.shape[0]
            total_items += noisy.shape[0]
            if step % 50 == 0:
                print(
                    f"  epoch {epoch} step {step}/{len(self.train_loader)} "
                    f"loss={float(loss):.5f}"
                )
        return total_loss / max(1, total_items)

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        self.model.eval()
        grouped: dict[tuple[str, str, tuple[int, ...]], list[dict[str, object]]] = defaultdict(list)
        for sample in self.val_samples:
            shape = tuple(sample["noisy"].shape)  # type: ignore[arg-type]
            grouped[(str(sample["dataset"]), str(sample["level"]), shape)].append(sample)
        scores: dict[tuple[str, str], list[float]] = defaultdict(list)
        for (dataset, level, _), samples in grouped.items():
            for start in range(0, len(samples), self.val_batch_size):
                batch = samples[start : start + self.val_batch_size]
                noisy = torch.stack([item["noisy"] for item in batch]).to(self.device)
                clean = torch.stack([item["clean"] for item in batch]).to(self.device)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                    prediction = denoise_full_image(self.model, noisy)
                prediction = prediction.float().clamp(0.0, 1.0)
                for value in _psnr(prediction, clean):
                    scores[(dataset, level)].append(value)
        output: dict[str, float] = {}
        pooled: dict[str, list[float]] = defaultdict(list)
        for (dataset, level), values in scores.items():
            finite = [value for value in values if math.isfinite(value)]
            output[f"psnr_{dataset}_{level}"] = sum(finite) / len(finite) if finite else math.nan
            pooled[level].extend(finite)
        for level, values in pooled.items():
            output[f"psnr_{level}"] = sum(values) / len(values) if values else math.nan
        noisy_values = [
            output[key]
            for key in ("psnr_low", "psnr_medium", "psnr_high")
            if key in output and math.isfinite(output[key])
        ]
        output["val_psnr"] = sum(noisy_values) / len(noisy_values) if noisy_values else math.nan
        return output

    def _append_history(self, row: dict[str, object]) -> None:
        self.history.append(row)
        fieldnames = list(row)
        temporary = self.history_path.with_suffix(".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)
        temporary.replace(self.history_path)

    def fit(self) -> dict[str, object]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        stale = 0
        best: dict[str, object] = {}
        for epoch in range(1, self.max_epochs + 1):
            started = time.perf_counter()
            train_loss = self._train_epoch(epoch)
            validation = self._validate()
            self.scheduler.step()
            row: dict[str, object] = {
                "epoch": epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                **validation,
                "seconds": time.perf_counter() - started,
            }
            self._append_history(row)
            score = float(validation["val_psnr"])
            improved = self.checkpoint.save(
                {
                    "model_state": self.model.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "scheduler_state": self.scheduler.state_dict(),
                    "config": self.config,
                    "seed": self.seed,
                },
                score,
                epoch,
                min_delta=self.min_delta,
            )
            stale = 0 if improved else stale + 1
            print(
                f"epoch {epoch:02d} loss={train_loss:.5f} val_psnr={score:.3f} "
                f"best={self.checkpoint.best_score:.3f} stale={stale}"
            )
            if improved:
                best = {
                    "epoch": epoch,
                    "val_psnr": score,
                    "validation": validation,
                    "train_loss": train_loss,
                }
            if stale >= self.patience:
                print(f"early stopping at epoch {epoch}")
                break
        return {
            "best": best,
            "best_epoch": self.checkpoint.best_epoch,
            "best_score": self.checkpoint.best_score,
            "epochs_ran": len(self.history),
            "history": self.history,
        }
