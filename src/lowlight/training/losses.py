"""Loss functions for CNN denoising."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class L1Loss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(prediction, target)


class L1PerceptualLoss(nn.Module):
    """L1 plus frozen AlexNet LPIPS in ``[-1, 1]``."""

    def __init__(self, lambda_lpips: float = 0.05, device: str | torch.device = "cuda") -> None:
        super().__init__()
        if lambda_lpips < 0.0:
            raise ValueError("lambda_lpips must be non-negative")
        import lpips

        self.lambda_lpips = float(lambda_lpips)
        self.perceptual = lpips.LPIPS(net="alex").to(device).eval()
        for parameter in self.perceptual.parameters():
            parameter.requires_grad_(False)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = F.l1_loss(prediction, target)
        if self.lambda_lpips == 0.0:
            return loss
        perceptual = self.perceptual(prediction * 2.0 - 1.0, target * 2.0 - 1.0)
        return loss + self.lambda_lpips * perceptual.mean()


def build_loss(
    name: str, *, lambda_lpips: float = 0.0, device: str | torch.device = "cuda"
) -> nn.Module:
    if name == "l1":
        return L1Loss()
    if name == "l1_perceptual":
        return L1PerceptualLoss(lambda_lpips=lambda_lpips, device=device)
    raise ValueError(f"unsupported loss: {name!r}")
