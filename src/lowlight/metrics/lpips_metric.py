"""LPIPS metric backed by the official ``lpips`` package."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from lowlight.metrics.base import BaseMetric, MetricInputError, as_image_pair


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LpipsMetric(BaseMetric):
    """Learned perceptual similarity; lower is better.

    Inputs are validated in ``[0, 1]`` and converted to ``[-1, 1]`` as
    required by the package API.
    """

    name = "lpips"
    lower_better = True

    def __init__(
        self,
        net: str = "alex",
        weights_version: str = "0.1",
        device: str | torch.device | None = None,
    ) -> None:
        import lpips

        self.net_name = net
        self.weights_version = weights_version
        self.device = resolve_device(device)
        self.model = lpips.LPIPS(
            net=net, version=weights_version, verbose=False
        ).to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def compute(self, prediction: object, target: object | None) -> float:
        if target is None:
            raise ValueError("LpipsMetric requires a reference target")
        pred, ref = as_image_pair(prediction, target)
        if pred.ndim != 3 or pred.shape[2] != 3:
            raise MetricInputError("lpips: expected RGB image with 3 channels")

        def tensor(image: np.ndarray) -> torch.Tensor:
            chw = np.transpose(image, (2, 0, 1)).astype(np.float32)
            return torch.from_numpy(chw).unsqueeze(0).to(self.device) * 2.0 - 1.0

        with (
            torch.inference_mode(),
            torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True),
        ):
            score = self.model(tensor(pred), tensor(ref)).mean()
        return float(score.detach().cpu().item())

    def weights_path(self) -> Path | None:
        import lpips

        package = Path(lpips.__file__).resolve().parent
        candidate = (
            package / "weights" / f"v{self.weights_version}" / f"{self.net_name}.pth"
        )
        return candidate if candidate.exists() else None

    def metadata(self) -> dict[str, object]:
        path = self.weights_path()
        backbone = sorted(
            Path.home().glob(".cache/torch/hub/checkpoints/alexnet-*.pth")
        )
        return {
            **super().metadata(),
            "package": "lpips",
            "package_version": version("lpips"),
            "backbone": self.net_name,
            "weights_version": self.weights_version,
            "weights_sha256": sha256_file(path) if path else None,
            "backbone_weights_sha256": sha256_file(backbone[-1]) if backbone else None,
            "device": str(self.device),
            "scaled_range": [-1.0, 1.0],
        }
