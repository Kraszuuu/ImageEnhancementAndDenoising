"""Classical brighteners with a shared interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import ClassVar

import cv2
import numpy as np
from skimage.color import lab2rgb, rgb2lab
from skimage.exposure import equalize_adapthist

from lowlight.metrics.base import as_image

EPSILON = 1e-6


def _squeeze(image: np.ndarray) -> tuple[np.ndarray, bool]:
    if image.ndim == 3 and image.shape[2] == 1:
        return image[..., 0], True
    return image, False


def _restore(image: np.ndarray, squeezed: bool) -> np.ndarray:
    if squeezed:
        return image[..., np.newaxis]
    return image


class BaseBrightener(ABC):
    """Common brightener interface. Inputs and outputs are RGB/gray in ``[0, 1]``."""

    name: ClassVar[str] = "brightener"

    def __init__(self, **params: object) -> None:
        self.params = params

    def enhance(self, image: object) -> np.ndarray:
        clean = as_image(image)
        output = self._enhance(clean)
        if output.shape != clean.shape:
            raise ValueError(
                f"{self.name}: output shape {output.shape} != input {clean.shape}"
            )
        if not np.all(np.isfinite(output)):
            raise ValueError(f"{self.name}: produced NaN or Inf")
        return np.clip(output.astype(np.float64, copy=False), 0.0, 1.0)

    @abstractmethod
    def _enhance(self, image: np.ndarray) -> np.ndarray:
        """Enhance a validated float64 image without mutating it."""

    def metadata(self) -> dict[str, object]:
        return {"name": self.name, "params": dict(self.params)}


class GammaBrightener(BaseBrightener):
    name = "gamma"

    def __init__(self, gamma: float) -> None:
        gamma = float(gamma)
        if gamma <= 0.0 or not np.isfinite(gamma):
            raise ValueError("gamma must be a positive finite number")
        super().__init__(gamma=gamma)

    def _enhance(self, image: np.ndarray) -> np.ndarray:
        gamma = float(self.params["gamma"])
        if gamma == 1.0:
            return image.copy()
        return np.power(image, gamma)


class ClaheBrightener(BaseBrightener):
    name = "clahe"

    def __init__(self, clip_limit: float, tile_grid_size: int = 8) -> None:
        clip_limit = float(clip_limit)
        tile_grid_size = int(tile_grid_size)
        if clip_limit <= 0.0 or not np.isfinite(clip_limit):
            raise ValueError("clip_limit must be a positive finite number")
        if tile_grid_size < 1:
            raise ValueError("tile_grid_size must be a positive integer")
        super().__init__(
            clip_limit=clip_limit,
            tile_grid_size=tile_grid_size,
        )

    def _enhance(self, image: np.ndarray) -> np.ndarray:
        array, squeezed = _squeeze(image)
        kernel_size = (int(self.params["tile_grid_size"]),) * 2
        clip_limit = float(self.params["clip_limit"])
        if array.ndim == 2:
            output = equalize_adapthist(
                array,
                kernel_size=kernel_size,
                clip_limit=clip_limit,
                nbins=256,
            )
            return _restore(output.astype(np.float64), squeezed)

        lab = rgb2lab(array)
        lightness = np.clip(lab[..., 0] / 100.0, 0.0, 1.0)
        lab[..., 0] = (
            equalize_adapthist(
                lightness,
                kernel_size=kernel_size,
                clip_limit=clip_limit,
                nbins=256,
            )
            * 100.0
        )
        return _restore(lab2rgb(lab).astype(np.float64), squeezed)


class MsrcrBrightener(BaseBrightener):
    name = "msrcr"

    def __init__(
        self,
        scales: Sequence[float],
        gain: float = 2.0,
        offset: float = 0.25,
        color_restoration: bool = True,
        normalize: bool = True,
    ) -> None:
        scales = tuple(float(value) for value in scales)
        if not scales:
            raise ValueError("scales must contain at least one value")
        if any(scale <= 0.0 or not np.isfinite(scale) for scale in scales):
            raise ValueError("scales must be positive finite numbers")
        gain = float(gain)
        offset = float(offset)
        if gain <= 0.0 or not np.isfinite(gain):
            raise ValueError("gain must be a positive finite number")
        if not np.isfinite(offset):
            raise ValueError("offset must be finite")
        super().__init__(
            scales=scales,
            gain=gain,
            offset=offset,
            color_restoration=bool(color_restoration),
            normalize=bool(normalize),
        )

    def _enhance(self, image: np.ndarray) -> np.ndarray:
        array, squeezed = _squeeze(image)
        scales = tuple(float(value) for value in self.params["scales"])
        gain = float(self.params["gain"])
        offset = float(self.params["offset"])
        use_color = bool(self.params["color_restoration"])
        normalize = bool(self.params["normalize"])
        weights = np.full(len(scales), 1.0 / len(scales), dtype=np.float64)

        log_image = np.log(np.clip(array, EPSILON, None) + EPSILON)
        msr = np.zeros_like(log_image)
        for scale, weight in zip(scales, weights, strict=True):
            blur = _gaussian_blur(log_image, scale)
            msr += weight * (log_image - blur)

        if use_color and array.ndim == 3 and array.shape[2] == 3:
            intensity = np.sum(array, axis=2, keepdims=True) + EPSILON
            msr *= array / intensity

        if normalize:
            low = float(msr.min())
            high = float(msr.max())
            msr = (msr - low) / (high - low + EPSILON)

        output = gain * msr + offset
        return _restore(output.astype(np.float64), squeezed)


def _gaussian_blur(array: np.ndarray, sigma: float) -> np.ndarray:
    """Approximate large Gaussian blurs on a downsampled image."""
    if sigma <= 8.0:
        return cv2.GaussianBlur(
            array,
            (0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
            borderType=cv2.BORDER_REFLECT,
        )
    height, width = array.shape[:2]
    factor = max(1, int(sigma // 8.0))
    factor = min(factor, max(1, min(height, width) // 8))
    small_width = max(1, width // factor)
    small_height = max(1, height // factor)
    small = cv2.resize(
        array, (small_width, small_height), interpolation=cv2.INTER_AREA
    )
    small = cv2.GaussianBlur(
        small,
        (0, 0),
        sigmaX=sigma / factor,
        sigmaY=sigma / factor,
        borderType=cv2.BORDER_REFLECT,
    )
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


BRIGHTENERS: dict[str, type[BaseBrightener]] = {
    "gamma": GammaBrightener,
    "clahe": ClaheBrightener,
    "msrcr": MsrcrBrightener,
}


def build_brightener(name: str, params: dict[str, object]) -> BaseBrightener:
    try:
        cls = BRIGHTENERS[name]
    except KeyError as error:
        raise ValueError(
            f"unknown brightener: {name!r}; available: {sorted(BRIGHTENERS)}"
        ) from error
    return cls(**params)
