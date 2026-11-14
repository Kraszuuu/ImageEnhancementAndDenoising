"""Shared metric interfaces and input validation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np

RANGE_TOLERANCE = 1e-9


class MetricInputError(ValueError):
    """Raised when an image is not a valid metric input."""


def as_image(array: object, *, name: str = "image") -> np.ndarray:
    """Validate a single image and return it as float64 in ``[0, 1]``.

    Accepted shapes are ``(H, W)``, ``(H, W, 1)`` and ``(H, W, 3)``.
    Values outside ``[0, 1]``, NaN, Inf and non-numeric dtypes are rejected.
    """
    image = np.asarray(array)
    if not np.issubdtype(image.dtype, np.number) or np.issubdtype(image.dtype, np.bool_):
        raise MetricInputError(f"{name}: expected numeric array, got {image.dtype}")
    if image.ndim not in (2, 3):
        raise MetricInputError(f"{name}: expected 2D or 3D array, got shape {image.shape}")
    if image.ndim == 3 and image.shape[2] not in (1, 3):
        raise MetricInputError(f"{name}: expected 1 or 3 channels, got {image.shape[2]}")
    if image.size == 0:
        raise MetricInputError(f"{name}: empty array")
    if not np.all(np.isfinite(image)):
        raise MetricInputError(f"{name}: contains NaN or Inf")
    converted = image.astype(np.float64, copy=False)
    low = float(converted.min())
    high = float(converted.max())
    if low < -RANGE_TOLERANCE or high > 1.0 + RANGE_TOLERANCE:
        raise MetricInputError(f"{name}: values outside [0, 1]: min={low}, max={high}")
    return np.clip(converted, 0.0, 1.0)


def as_image_pair(
    prediction: object,
    target: object,
    *,
    prediction_name: str = "prediction",
    target_name: str = "target",
) -> tuple[np.ndarray, np.ndarray]:
    pred = as_image(prediction, name=prediction_name)
    ref = as_image(target, name=target_name)
    if pred.shape != ref.shape:
        raise MetricInputError(f"shape mismatch: {pred.shape} != {ref.shape}")
    return pred, ref


class BaseMetric(ABC):
    """Common interface for image quality metrics."""

    name: ClassVar[str] = "metric"
    lower_better: ClassVar[bool] = False

    @abstractmethod
    def compute(self, prediction: object, target: object | None) -> float:
        """Return a scalar score. Full-reference metrics require ``target``."""

    def metadata(self) -> dict[str, object]:
        return {"name": self.name, "lower_better": self.lower_better}
