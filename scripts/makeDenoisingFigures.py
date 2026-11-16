"""Create Stage 03 validation curves and qualitative test panels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lowlight.classical.denoisers import build_denoiser  # noqa: E402
from lowlight.metrics import PsnrMetric  # noqa: E402
from lowlight.models import build_model, denoise_full_image  # noqa: E402
from lowlight.training.checkpointing import CheckpointManager  # noqa: E402
from lowlight.utils.io import load_rgb, read_csv_rows  # noqa: E402
from lowlight.utils.noise import load_noise_config  # noqa: E402

GAMMA = 0.45
EXAMPLE = "data/raw/lolv2-real/Test/Input/00761.png"


def display(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0) ** GAMMA


def load_cnn(seed: int, device: torch.device):
    path = ROOT / "checkpoints" / "denoising_cnn" / f"seed{seed}" / "best.pt"
    payload = torch.load(path, map_location=device, weights_only=False)
    model = build_model(payload["config"]["model"]).to(device).eval()
    CheckpointManager.load(path, model, device=device)
    model.eval()
    return model


def predict(model, noisy: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(noisy).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        output = denoise_full_image(model, tensor)
    return output[0].permute(1, 2, 0).float().clamp(0.0, 1.0).cpu().numpy()


def validation_curves(out_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for seed in (0, 1, 2):
        rows = read_csv_rows(ROOT / "results" / "denoising_cnn" / f"seed{seed}" / "history.csv")
        epochs = [int(row["epoch"]) for row in rows]
        axes[0].plot(epochs, [float(row["train_loss"]) for row in rows], label=f"seed {seed}")
        axes[1].plot(epochs, [float(row["val_psnr"]) for row in rows], label=f"seed {seed}")
    axes[0].set_title("Training loss (L1 + 0.05 LPIPS)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[1].set_title("Validation PSNR (low+medium+high)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("dB")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(out_dir / "val_curves.png", dpi=150)
    plt.close(figure)


def qualitative(out_dir: Path, device: torch.device) -> None:
    clean = load_rgb(ROOT / EXAMPLE)
    noisy = load_noise_config(ROOT / "configs" / "data" / "noise.yaml").add(
        clean, "high", image_id=EXAMPLE
    )
    dfixed = build_denoiser("bm3d", {"sigma_psd": 0.12, "profile": "np", "num_threads": 0})
    outputs = {
        "L target": clean,
        "L+n high": noisy,
        "CNN seed 0": predict(load_cnn(0, device), noisy, device),
        "CNN seed 1": predict(load_cnn(1, device), noisy, device),
        "CNN seed 2": predict(load_cnn(2, device), noisy, device),
        "D_fixed BM3D": dfixed.denoise(noisy),
    }
    metric = PsnrMetric()
    figure, axes = plt.subplots(2, 3, figsize=(12, 6.6))
    for axis, (label, image) in zip(axes.ravel(), outputs.items(), strict=True):
        axis.imshow(display(image))
        score = "" if label == "L target" else f"\nPSNR {metric.compute(image, clean):.2f} dB"
        axis.set_title(f"{label}{score}", fontsize=9)
        axis.axis("off")
    figure.suptitle("Stage 03 qualitative result, high noise, LOL-v2 Real 00761", fontsize=10)
    figure.tight_layout()
    figure.savefig(out_dir / "qualitative_high.png", dpi=150)
    plt.close(figure)


def worst_images() -> dict[str, str]:
    cnn = read_csv_rows(ROOT / "results" / "denoising_cnn" / "evaluation" / "per_image.csv")
    dfixed = read_csv_rows(
        ROOT / "results" / "denoising_cnn" / "evaluation_d_fixed" / "per_image.csv"
    )
    cnn_values: dict[tuple[str, str, str], list[float]] = {}
    for row in cnn:
        if row["noise_level"] != "high":
            continue
        key = (row["dataset"], row["image_id"], row["noise_level"])
        cnn_values.setdefault(key, []).append(float(row["psnr"]))
    reference = {
        (row["dataset"], row["image_id"], row["noise_level"]): float(row["psnr"])
        for row in dfixed
        if row["noise_level"] == "high"
    }
    selected = {}
    for key, values in cnn_values.items():
        if key not in reference:
            continue
        delta = float(np.mean(values)) - reference[key]
        if key[0] not in selected or delta < selected[key[0]][0]:
            selected[key[0]] = (delta, key[1])
    return {dataset: image_id for dataset, (_, image_id) in selected.items()}


def failures(out_dir: Path, device: torch.device) -> None:
    dfixed = build_denoiser("bm3d", {"sigma_psd": 0.12, "profile": "np", "num_threads": 0})
    selected = worst_images()
    metric = PsnrMetric()
    figure, axes = plt.subplots(len(selected), 4, figsize=(12, 3.2 * len(selected)), squeeze=False)
    for row_index, (dataset, image_id) in enumerate(sorted(selected.items())):
        clean = load_rgb(ROOT / image_id)
        noisy = load_noise_config(ROOT / "configs" / "data" / "noise.yaml").add(
            clean, "high", image_id=image_id
        )
        cnn = predict(load_cnn(0, device), noisy, device)
        fixed = dfixed.denoise(noisy)
        panels = [
            ("target", clean),
            ("noisy high", noisy),
            ("CNN seed 0", cnn),
            ("D_fixed", fixed),
        ]
        for axis, (label, image) in zip(axes[row_index], panels, strict=True):
            axis.imshow(display(image))
            score = (
                ""
                if label in {"target", "noisy high"}
                else f" {metric.compute(image, clean):.2f} dB"
            )
            axis.set_title(f"{dataset} {Path(image_id).stem} {label}{score}", fontsize=8)
            axis.axis("off")
    figure.suptitle("Failure cases: largest CNN - D_fixed PSNR gap at high noise", fontsize=10)
    figure.tight_layout()
    figure.savefig(out_dir / "failure_cases_high.png", dpi=150)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "results" / "denoising_cnn" / "figures"
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    validation_curves(args.out_dir)
    qualitative(args.out_dir, device)
    failures(args.out_dir, device)
    print(f"figures={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
