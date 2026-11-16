"""Reusable convolutional blocks for the compact U-Net."""

from __future__ import annotations

import torch
from torch import nn


def norm_groups(channels: int, requested: int = 8) -> int:
    for groups in range(min(requested, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        groups: int = 8,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(norm_groups(out_channels, groups), out_channels),
            activation(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(norm_groups(out_channels, groups), out_channels),
            activation(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        groups: int = 8,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = ConvBlock(
            in_channels, out_channels, groups=groups, activation=activation
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(inputs))


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        *,
        groups: int = 8,
        activation: type[nn.Module] = nn.SiLU,
    ) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = ConvBlock(
            out_channels + skip_channels,
            out_channels,
            groups=groups,
            activation=activation,
        )

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upsampled = self.up(inputs)
        if upsampled.shape[-2:] != skip.shape[-2:]:
            upsampled = nn.functional.interpolate(
                upsampled, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.conv(torch.cat([skip, upsampled], dim=1))
