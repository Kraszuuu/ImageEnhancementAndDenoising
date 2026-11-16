"""Classical brightening pipeline with an optional frozen denoiser."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lowlight.classical.brighteners import BaseBrightener
from lowlight.classical.denoisers import BaseDenoiser


@dataclass
class ClassicalBrighteningPipeline:
    brightener: BaseBrightener
    denoiser: BaseDenoiser | None = None

    def run(self, image: object) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(brightened, source)`` for one low-light image."""
        source = self.denoiser.denoise(image) if self.denoiser is not None else image
        return self.brightener.enhance(source), np.asarray(source)

    def enhance(self, image: object) -> np.ndarray:
        brightened, _ = self.run(image)
        return brightened

    def metadata(self) -> dict[str, object]:
        return {
            "brightener": self.brightener.metadata(),
            "denoiser": self.denoiser.metadata() if self.denoiser is not None else None,
        }
