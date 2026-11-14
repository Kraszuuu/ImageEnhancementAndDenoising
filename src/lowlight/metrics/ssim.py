"""SSIM metric frozen to the scikit-image reference implementation."""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity

from lowlight.metrics.base import BaseMetric, as_image_pair

SSIM_KWARGS = {
    "data_range": 1.0,
    "gaussian_weights": True,
    "sigma": 1.5,
    "use_sample_covariance": False,
}


def ssim(prediction: np.ndarray, target: np.ndarray) -> float:
    """SSIM with the frozen protocol: mean over RGB channels for color images."""
    if prediction.ndim == 3 and prediction.shape[2] == 1:
        prediction = prediction[..., 0]
        target = target[..., 0]
    channel_axis = 2 if prediction.ndim == 3 else None
    return float(
        structural_similarity(
            target,
            prediction,
            channel_axis=channel_axis,
            **SSIM_KWARGS,
        )
    )


class SsimMetric(BaseMetric):
    """Structural similarity index in ``[0, 1]`` (higher is better)."""

    name = "ssim"
    lower_better = False

    def compute(self, prediction: object, target: object | None) -> float:
        if target is None:
            raise ValueError("SsimMetric requires a reference target")
        pred, ref = as_image_pair(prediction, target)
        return ssim(pred, ref)

    def metadata(self) -> dict[str, object]:
        return {
            **super().metadata(),
            "implementation": "skimage.metrics.structural_similarity",
            **SSIM_KWARGS,
            "channel_axis": 2,
        }
