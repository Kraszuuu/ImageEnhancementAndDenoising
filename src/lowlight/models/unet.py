"""Compact U-Net used for CNN denoising and brightening experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from lowlight.models.blocks import ConvBlock, DownBlock, UpBlock


def _activation(name: str) -> type[nn.Module]:
    if name.lower() == "silu":
        return nn.SiLU
    if name.lower() == "relu":
        return nn.ReLU
    raise ValueError(f"unsupported activation: {name!r}")


@dataclass(frozen=True)
class UNetConfig:
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 32
    depth: int = 4
    groups: int = 8
    activation: str = "silu"
    output_activation: str = "clamp"
    residual: bool = False

    def __post_init__(self) -> None:
        if self.in_channels < 1 or self.out_channels < 1:
            raise ValueError("channel counts must be positive")
        if self.base_channels < 1:
            raise ValueError("base_channels must be positive")
        if self.depth < 2:
            raise ValueError("depth must be at least 2")
        if self.output_activation not in {"clamp", "sigmoid"}:
            raise ValueError("output_activation must be 'clamp' or 'sigmoid'")


class CompactUNet(nn.Module):
    """Four-level encoder-decoder with GroupNorm and skip connections."""

    def __init__(self, config: UNetConfig) -> None:
        super().__init__()
        self.config = config
        activation = _activation(config.activation)
        channels = [config.base_channels * 2**level for level in range(config.depth)]

        self.stem = ConvBlock(
            config.in_channels, channels[0], groups=config.groups, activation=activation
        )
        self.down = nn.ModuleList(
            [
                DownBlock(
                    channels[index - 1],
                    channels[index],
                    groups=config.groups,
                    activation=activation,
                )
                for index in range(1, config.depth)
            ]
        )
        self.bottleneck = DownBlock(
            channels[-1],
            channels[-1] * 2,
            groups=config.groups,
            activation=activation,
        )
        self.up = nn.ModuleList(
            [
                UpBlock(
                    (channels[-1] * 2) if index == config.depth - 1 else channels[index + 1],
                    channels[index],
                    channels[index],
                    groups=config.groups,
                    activation=activation,
                )
                for index in reversed(range(config.depth))
            ]
        )
        self.head = nn.Conv2d(channels[0], config.out_channels, kernel_size=1)

    @property
    def size_multiple(self) -> int:
        return 2**self.config.depth

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.config.in_channels:
            raise ValueError(
                f"expected [B, {self.config.in_channels}, H, W], got {tuple(inputs.shape)}"
            )
        if not torch.isfinite(inputs).all():
            raise ValueError("input contains NaN or Inf")
        multiple = self.size_multiple
        if inputs.shape[-2] % multiple or inputs.shape[-1] % multiple:
            raise ValueError(
                f"input height and width must be multiples of {multiple}; "
                f"got {tuple(inputs.shape[-2:])}"
            )

        skips = [self.stem(inputs)]
        for down in self.down:
            skips.append(down(skips[-1]))
        current = self.bottleneck(skips[-1])
        for up, skip in zip(self.up, reversed(skips), strict=True):
            current = up(current, skip)
        output = self.head(current)
        if self.config.residual:
            output = inputs + output
        if self.config.output_activation == "sigmoid":
            return torch.sigmoid(output)
        return torch.clamp(output, 0.0, 1.0)


def build_model(config: dict[str, object]) -> CompactUNet:
    model_config = UNetConfig(
        in_channels=int(config.get("in_channels", 3)),
        out_channels=int(config.get("out_channels", 3)),
        base_channels=int(config.get("base_channels", 32)),
        depth=int(config.get("depth", 4)),
        groups=int(config.get("groups", 8)),
        activation=str(config.get("activation", "silu")),
        output_activation=str(config.get("output_activation", "clamp")),
        residual=bool(config.get("residual", False)),
    )
    return CompactUNet(model_config)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def pad_to_multiple(
    tensor: torch.Tensor, multiple: int, mode: str = "reflect"
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Pad the last two dimensions and return the crop needed to undo it."""
    if multiple < 1:
        raise ValueError("multiple must be positive")
    height, width = tensor.shape[-2:]
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    if pad_height == 0 and pad_width == 0:
        return tensor, (0, 0, 0, 0)
    padded = F.pad(tensor, (0, pad_width, 0, pad_height), mode=mode)
    return padded, (0, pad_width, 0, pad_height)


def unpad(tensor: torch.Tensor, padding: tuple[int, int, int, int]) -> torch.Tensor:
    left, right, top, bottom = padding
    height, width = tensor.shape[-2:]
    return tensor[..., top : height - bottom, left : width - right]


def denoise_full_image(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Run a full image through the model with reflection padding."""
    if image.ndim == 3:
        image = image.unsqueeze(0)
    multiple = getattr(model, "size_multiple", 16)
    padded, padding = pad_to_multiple(image, multiple)
    output = model(padded)
    return unpad(output, padding)

