"""Learning-rate schedules for the denoising trainer."""

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosine(LRScheduler):
    """Linear warmup followed by cosine decay to ``eta_min``."""

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        total_epochs: int,
        warmup_epochs: int = 0,
        eta_min_factor: float = 0.01,
        last_epoch: int = -1,
    ) -> None:
        if total_epochs < 1:
            raise ValueError("total_epochs must be positive")
        if warmup_epochs < 0 or warmup_epochs >= total_epochs:
            raise ValueError("warmup_epochs must be in [0, total_epochs)")
        if not 0.0 <= eta_min_factor < 1.0:
            raise ValueError("eta_min_factor must be in [0, 1)")
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.eta_min_factor = eta_min_factor
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_epochs:
            scale = (self.last_epoch + 1) / max(1, self.warmup_epochs)
        else:
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            scale = self.eta_min_factor + (1.0 - self.eta_min_factor) * 0.5 * (
                1.0 + math.cos(math.pi * min(progress, 1.0))
            )
        return [base_lr * scale for base_lr in self.base_lrs]


def build_scheduler(
    optimizer: Optimizer,
    *,
    name: str,
    total_epochs: int,
    warmup_epochs: int = 0,
    eta_min_factor: float = 0.01,
) -> LRScheduler:
    if name != "cosine":
        raise ValueError(f"unsupported scheduler: {name!r}")
    return WarmupCosine(
        optimizer,
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        eta_min_factor=eta_min_factor,
    )
