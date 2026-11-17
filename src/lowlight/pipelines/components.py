"""Components that adapt classical and neural models to the pipeline API."""

from __future__ import annotations

import time

import numpy as np
import torch

from lowlight.classical.brighteners import BaseBrightener
from lowlight.classical.denoisers import BaseDenoiser
from lowlight.metrics.base import as_image
from lowlight.models import count_parameters, denoise_full_image


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class ClassicalDenoiserComponent:
    def __init__(self, denoiser: BaseDenoiser) -> None:
        self.denoiser = denoiser

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.denoiser.denoise(image)

    def metadata(self) -> dict[str, object]:
        payload = self.denoiser.metadata()
        return {"kind": "classical_denoiser", **payload}


class ClassicalBrightenerComponent:
    def __init__(self, brightener: BaseBrightener) -> None:
        self.brightener = brightener

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.brightener.enhance(image)

    def metadata(self) -> dict[str, object]:
        payload = self.brightener.metadata()
        return {"kind": "classical_brightener", **payload}


class NeuralComponent:
    """Run a trained RGB U-Net on a full image with reflection padding."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: str | torch.device = "cpu",
        name: str = "cnn",
        checkpoint: str | None = None,
        seed: int | None = None,
        epoch: int | None = None,
        parameters: int | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.name = name
        self.checkpoint = checkpoint
        self.seed = seed
        self.epoch = epoch
        self.parameters = count_parameters(model) if parameters is None else parameters

    def __call__(self, image: np.ndarray) -> np.ndarray:
        array = as_image(image, name=self.name)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"{self.name}: expected RGB image, got shape {array.shape}")
        tensor = (
            torch.from_numpy(np.ascontiguousarray(array))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )
        _synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = denoise_full_image(self.model, tensor)
        _synchronize(self.device)
        self.last_ms = (time.perf_counter() - started) * 1000.0
        return output[0].permute(1, 2, 0).float().clamp(0.0, 1.0).cpu().numpy().astype(np.float64)

    def metadata(self) -> dict[str, object]:
        return {
            "kind": "neural",
            "name": self.name,
            "device": str(self.device),
            "checkpoint": self.checkpoint,
            "seed": self.seed,
            "epoch": self.epoch,
            "parameters": self.parameters,
        }
