"""NIQE no-reference metric backed by ``pyiqa``."""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from lowlight.metrics.base import BaseMetric, MetricInputError, as_image
from lowlight.metrics.lpips_metric import resolve_device, sha256_file

MIN_SIDE = 96


class NiqeMetric(BaseMetric):
    """Natural Image Quality Evaluator; lower is better.

    The score is no-reference, so ``target`` is ignored and may be ``None``.
    """

    name = "niqe"
    lower_better = True

    def __init__(
        self,
        device: str | torch.device | None = None,
        weights_version: str = "original",
    ) -> None:
        import pyiqa

        self.device = resolve_device(device)
        self.weights_version = weights_version
        self.model = pyiqa.create_metric(
            "niqe", device=str(self.device), version=weights_version
        )

    def compute(self, prediction: object, target: object | None) -> float:
        del target
        image = as_image(prediction, name="prediction")
        height, width = image.shape[:2]
        if height < MIN_SIDE or width < MIN_SIDE:
            raise MetricInputError(
                f"niqe: minimum side is {MIN_SIDE} px, got {height}x{width}"
            )
        if image.ndim == 2:
            array = image[np.newaxis, :, :]
        elif image.shape[2] == 1:
            array = np.transpose(image[..., 0], (0, 1))[np.newaxis, :, :]
        else:
            array = np.transpose(image, (2, 0, 1))
        tensor = torch.from_numpy(array.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            score = self.model(tensor)
        return float(score.mean().detach().cpu().item())

    def params_path(self) -> Path | None:
        name = (
            "niqe_modelparameters.mat"
            if self.weights_version == "original"
            else "niqe_matlab_params.mat"
        )
        candidate = Path(torch.hub.get_dir()) / "pyiqa" / name
        return candidate if candidate.exists() else None

    def metadata(self) -> dict[str, object]:
        path = self.params_path()
        return {
            **super().metadata(),
            "package": "pyiqa",
            "package_version": version("pyiqa"),
            "metric": "niqe",
            "weights_version": self.weights_version,
            "params_sha256": sha256_file(path) if path else None,
            "device": str(self.device),
            "min_side": MIN_SIDE,
            "test_y_channel": True,
            "reference_required": False,
        }
