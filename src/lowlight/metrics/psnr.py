"""PSNR metric with RGB and luminance variants."""

from __future__ import annotations

import math

import numpy as np

from lowlight.metrics.base import BaseMetric, as_image_pair

LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)


def to_luminance(image: np.ndarray) -> np.ndarray:
    """Rec.709 luminance for RGB input; grayscale input is returned unchanged."""
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[..., 0]
    weights = np.asarray(LUMA_WEIGHTS, dtype=np.float64)
    return image @ weights


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction - target) ** 2))


def psnr_from_mse(mse_value: float, data_range: float = 1.0) -> float:
    if mse_value < 0.0:
        raise ValueError(f"MSE must be non-negative, got {mse_value}")
    if mse_value == 0.0:
        return math.inf
    return float(10.0 * math.log10(data_range**2 / mse_value))


class PsnrMetric(BaseMetric):
    """Peak signal-to-noise ratio in dB, computed on RGB or luminance."""

    name = "psnr"
    lower_better = False

    def __init__(self, channel: str = "rgb") -> None:
        if channel not in {"rgb", "luma"}:
            raise ValueError(f"channel must be 'rgb' or 'luma', got {channel!r}")
        self.channel = channel

    def compute(self, prediction: object, target: object | None) -> float:
        if target is None:
            raise ValueError("PsnrMetric requires a reference target")
        pred, ref = as_image_pair(prediction, target)
        if self.channel == "luma":
            pred = to_luminance(pred)
            ref = to_luminance(ref)
        return psnr_from_mse(mse(pred, ref))

    def metadata(self) -> dict[str, object]:
        return {
            **super().metadata(),
            "channel": self.channel,
            "luma_weights": list(LUMA_WEIGHTS),
            "mse_zero": "inf",
        }
