"""Classical denoising pipeline: add controlled noise, then denoise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lowlight.classical.denoisers import BaseDenoiser
from lowlight.utils.noise import NoiseConfig


@dataclass
class ClassicalDenoisingPipeline:
    denoiser: BaseDenoiser
    noise_config: NoiseConfig
    seed: int = 0

    def run(
        self, image: object, level: str, *, image_id: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(denoised, noisy)`` for one image and noise level."""
        noisy = self.noise_config.add(
            image, level, image_id=image_id, seed=self.seed
        )
        return self.denoiser.denoise(noisy), noisy

    def denoise(
        self, image: object, level: str, *, image_id: str
    ) -> np.ndarray:
        denoised, _ = self.run(image, level, image_id=image_id)
        return denoised

    def metadata(self) -> dict[str, object]:
        return {
            "denoiser": self.denoiser.metadata(),
            "noise_levels": {
                name: {"alpha": level.alpha, "sigma": level.sigma}
                for name, level in self.noise_config.levels.items()
            },
            "seed": self.seed,
        }
