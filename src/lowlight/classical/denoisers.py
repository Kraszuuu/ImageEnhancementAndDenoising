"""Classical denoisers with a shared interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage.restoration import denoise_nl_means

from lowlight.metrics.base import as_image


def _squeeze(image: np.ndarray) -> tuple[np.ndarray, bool]:
    if image.ndim == 3 and image.shape[2] == 1:
        return image[..., 0], True
    return image, False


def _restore(image: np.ndarray, squeezed: bool) -> np.ndarray:
    if squeezed:
        return image[..., np.newaxis]
    return image


class BaseDenoiser(ABC):
    """Common denoiser interface. Inputs and outputs are RGB/gray in ``[0, 1]``."""

    name: ClassVar[str] = "denoiser"

    def __init__(self, **params: object) -> None:
        self.params = params

    def denoise(self, image: object) -> np.ndarray:
        clean = as_image(image)
        output = self._denoise(clean)
        if output.shape != clean.shape:
            raise ValueError(
                f"{self.name}: output shape {output.shape} != input {clean.shape}"
            )
        if not np.all(np.isfinite(output)):
            raise ValueError(f"{self.name}: produced NaN or Inf")
        return np.clip(output.astype(np.float64, copy=False), 0.0, 1.0)

    @abstractmethod
    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Denoise a validated float64 image without mutating it."""

    def metadata(self) -> dict[str, object]:
        return {"name": self.name, "params": dict(self.params)}


class GaussianDenoiser(BaseDenoiser):
    name = "gaussian"

    def __init__(self, sigma: float) -> None:
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        super().__init__(sigma=float(sigma))

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        sigma = float(self.params["sigma"])
        if image.ndim == 3:
            return gaussian_filter(image, sigma=(sigma, sigma, 0.0))
        return gaussian_filter(image, sigma=sigma)


class MedianDenoiser(BaseDenoiser):
    name = "median"

    def __init__(self, kernel_size: int) -> None:
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        super().__init__(kernel_size=int(kernel_size))

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        size = int(self.params["kernel_size"])
        if image.ndim == 3:
            return median_filter(image, size=(size, size, 1))
        return median_filter(image, size=size)


class BilateralDenoiser(BaseDenoiser):
    name = "bilateral"

    def __init__(self, d: int, sigma_color: float, sigma_space: float) -> None:
        if d < 1 or d % 2 == 0:
            raise ValueError("d must be a positive odd integer")
        if sigma_color <= 0 or sigma_space <= 0:
            raise ValueError("sigma_color and sigma_space must be positive")
        super().__init__(
            d=int(d), sigma_color=float(sigma_color), sigma_space=float(sigma_space)
        )

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        squeezed = image.ndim == 3 and image.shape[2] == 1
        array, _ = _squeeze(image)
        output = cv2.bilateralFilter(
            array.astype(np.float32),
            int(self.params["d"]),
            float(self.params["sigma_color"]),
            float(self.params["sigma_space"]),
        )
        return _restore(output.astype(np.float64), squeezed)


class NlMeansDenoiser(BaseDenoiser):
    name = "nlm"

    def __init__(
        self,
        h: float,
        patch_size: int = 7,
        patch_distance: int = 11,
        fast_mode: bool = True,
    ) -> None:
        if h <= 0:
            raise ValueError("h must be positive")
        if patch_size < 1 or patch_size % 2 == 0:
            raise ValueError("patch_size must be a positive odd integer")
        if patch_distance < 1:
            raise ValueError("patch_distance must be positive")
        super().__init__(
            h=float(h),
            patch_size=int(patch_size),
            patch_distance=int(patch_distance),
            fast_mode=bool(fast_mode),
        )

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        squeezed = image.ndim == 3 and image.shape[2] == 1
        array, _ = _squeeze(image)
        channel_axis = -1 if array.ndim == 3 else None
        output = denoise_nl_means(
            array,
            h=float(self.params["h"]),
            patch_size=int(self.params["patch_size"]),
            patch_distance=int(self.params["patch_distance"]),
            fast_mode=bool(self.params["fast_mode"]),
            channel_axis=channel_axis,
            preserve_range=False,
        )
        return _restore(np.asarray(output, dtype=np.float64), squeezed)


class Bm3dDenoiser(BaseDenoiser):
    name = "bm3d"

    def __init__(
        self, sigma_psd: float, profile: str = "np", num_threads: int = 0
    ) -> None:
        if sigma_psd <= 0:
            raise ValueError("sigma_psd must be positive")
        if profile not in {"np", "refilter", "vn", "vn_old", "high", "deb"}:
            raise ValueError(f"unsupported BM3D profile: {profile!r}")
        if num_threads < 0:
            raise ValueError("num_threads must be >= 0")
        super().__init__(
            sigma_psd=float(sigma_psd), profile=profile, num_threads=int(num_threads)
        )

    def _profile(self):
        import bm3d

        profiles = {
            "np": bm3d.BM3DProfile,
            "refilter": bm3d.BM3DProfileRefilter,
            "vn": bm3d.BM3DProfileVN,
            "vn_old": bm3d.BM3DProfileVNOld,
            "high": bm3d.BM3DProfileHigh,
            "deb": bm3d.BM3DProfileDeb,
        }
        profile = profiles[str(self.params["profile"])]()
        profile.num_threads = int(self.params["num_threads"])
        return profile

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        import bm3d

        squeezed = image.ndim == 3 and image.shape[2] == 1
        array, _ = _squeeze(image)
        sigma = float(self.params["sigma_psd"])
        profile = self._profile()
        if array.ndim == 3:
            output = bm3d.bm3d_rgb(array, sigma, profile=profile)
        else:
            output = bm3d.bm3d(array, sigma, profile=profile)
        return _restore(np.asarray(output, dtype=np.float64), squeezed)


DENOISERS: dict[str, type[BaseDenoiser]] = {
    "gaussian": GaussianDenoiser,
    "median": MedianDenoiser,
    "bilateral": BilateralDenoiser,
    "nlm": NlMeansDenoiser,
    "bm3d": Bm3dDenoiser,
}


def build_denoiser(name: str, params: dict[str, object]) -> BaseDenoiser:
    try:
        cls = DENOISERS[name]
    except KeyError as error:
        raise ValueError(
            f"unknown denoiser: {name!r}; available: {sorted(DENOISERS)}"
        ) from error
    return cls(**params)
