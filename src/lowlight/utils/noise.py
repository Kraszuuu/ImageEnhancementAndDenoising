"""Shared sRGB/linear-light noise model (Gaussian-Poisson, controlled)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from lowlight.metrics.base import as_image

LEVEL_NAMES = ("clean", "low", "medium", "high")


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    """Convert sRGB values in ``[0, 1]`` to linear light."""
    return np.where(
        image <= 0.04045,
        image / 12.92,
        ((image + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    """Convert linear-light values in ``[0, 1]`` to sRGB."""
    clipped = np.clip(image, 0.0, 1.0)
    return np.where(
        clipped <= 0.0031308,
        12.92 * clipped,
        1.055 * np.power(clipped, 1.0 / 2.4) - 0.055,
    )


def stable_seed(base_seed: int, image_id: str, level: str) -> int:
    """Derive a stable RNG seed from the run seed, image id and level."""
    payload = f"{base_seed}|{image_id}|{level}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


def add_noise(
    image: object,
    *,
    alpha: float,
    sigma: float,
    seed: int,
) -> np.ndarray:
    """Add Gaussian-Poisson noise in linear light and return sRGB.

    ``alpha`` controls shot noise, ``sigma`` controls read noise. Both are
    expressed in linear-light units. The input is never modified.
    """
    if alpha < 0.0 or sigma < 0.0:
        raise ValueError("alpha and sigma must be non-negative")
    clean = as_image(image)
    if alpha == 0.0 and sigma == 0.0:
        return clean.copy()

    linear = srgb_to_linear(clean)
    rng = np.random.default_rng(seed)
    shot = np.zeros_like(linear)
    if alpha > 0.0:
        shot = alpha * rng.poisson(linear / alpha).astype(np.float64)
    read = rng.normal(0.0, sigma, size=linear.shape) if sigma > 0.0 else 0.0
    noisy = np.clip(shot + read, 0.0, 1.0)
    return np.clip(linear_to_srgb(noisy), 0.0, 1.0)


@dataclass(frozen=True)
class NoiseLevel:
    alpha: float
    sigma: float


@dataclass(frozen=True)
class NoiseConfig:
    color_space: str
    range: tuple[float, float]
    apply_in: str
    seed_policy: str
    levels: dict[str, NoiseLevel]

    def level(self, name: str) -> NoiseLevel:
        try:
            return self.levels[name]
        except KeyError as error:
            raise ValueError(f"unknown noise level: {name!r}") from error

    def add(self, image: object, level: str, *, image_id: str, seed: int = 0) -> np.ndarray:
        if level not in LEVEL_NAMES:
            raise ValueError(f"unknown noise level: {level!r}")
        parameters = self.level(level)
        if parameters.alpha == 0.0 and parameters.sigma == 0.0:
            return as_image(image).copy()
        return add_noise(
            image,
            alpha=parameters.alpha,
            sigma=parameters.sigma,
            seed=stable_seed(seed, image_id, level),
        )


def load_noise_config(path: Path) -> NoiseConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    levels = {}
    for name in LEVEL_NAMES:
        raw = payload["levels"][name]
        alpha = raw.get("alpha")
        sigma = raw.get("sigma")
        if alpha is None or sigma is None:
            raise ValueError(
                f"noise level {name!r} is not calibrated in {path}: "
                "alpha and sigma must be numbers"
            )
        levels[name] = NoiseLevel(alpha=float(alpha), sigma=float(sigma))
    return NoiseConfig(
        color_space=str(payload["color_space"]),
        range=tuple(payload["range"]),
        apply_in=str(payload["apply_in"]),
        seed_policy=str(payload["seed_policy"]),
        levels=levels,
    )
