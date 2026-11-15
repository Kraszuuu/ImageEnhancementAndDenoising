"""Render qualitative classical-denoising figures from the frozen configuration."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lowlight.classical.denoisers import build_denoiser  # noqa: E402
from lowlight.utils.io import load_rgb, read_csv_rows  # noqa: E402
from lowlight.utils.noise import load_noise_config  # noqa: E402

LEVELS = ("clean", "low", "medium", "high")
METHODS = ("gaussian", "median", "bilateral", "nlm", "bm3d")


def display(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0) ** 0.45


def render(
    panels: list[tuple[str, np.ndarray]],
    path: Path,
    title: str,
    columns: int = 4,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = math.ceil(len(panels) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(2.8 * columns, 2.6 * rows))
    for axis in axes.ravel():
        axis.axis("off")
    for axis, (label, image) in zip(axes.ravel(), panels, strict=False):
        axis.imshow(display(image))
        axis.set_title(label, fontsize=8)
        axis.axis("off")
    figure.suptitle(f"{title} (display gamma 0.45)", fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def median_brightness_image() -> str:
    rows = read_csv_rows(ROOT / "data" / "splits" / "lolv2_test.csv")
    scored = sorted(
        (float(load_rgb(ROOT / row["low_path"]).mean()), row["low_path"])
        for row in rows
    )
    return scored[len(scored) // 2][1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen",
        type=Path,
        default=ROOT / "configs" / "classical" / "denoising_frozen.yaml",
    )
    parser.add_argument(
        "--noise-config", type=Path, default=ROOT / "configs" / "data" / "noise.yaml"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "results" / "classical_denoising" / "figures"
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frozen = yaml.safe_load(args.frozen.read_text(encoding="utf-8"))
    noise_config = load_noise_config(args.noise_config)
    denoisers = {
        (method, level): build_denoiser(method, frozen["methods"][method][level]["params"])
        for method in METHODS
        for level in LEVELS
    }

    image_id = median_brightness_image()
    clean = load_rgb(ROOT / image_id)
    noisy = {level: noise_config.add(clean, level, image_id=image_id) for level in LEVELS}
    outputs = {
        (method, level): denoisers[(method, level)].denoise(noisy[level])
        for method in METHODS
        for level in LEVELS
    }

    render(
        [("target L", clean)]
        + [(f"L+n {level}", noisy[level]) for level in ("low", "medium", "high")],
        args.out_dir / "noise_levels.png",
        f"Noise levels: {Path(image_id).name}",
        columns=4,
    )
    for level in LEVELS:
        source = [("target L", clean)]
        if level != "clean":
            source.append((f"L+n {level}", noisy[level]))
        source.extend((f"{method} {level}", outputs[(method, level)]) for method in METHODS)
        render(
            source,
            args.out_dir / f"denoise_{level}.png",
            f"Classical denoising, {level}",
            columns=4,
        )

    height, width = clean.shape[:2]
    crop = (slice(height // 3, height // 3 + 160), slice(width // 3, width // 3 + 240))
    render(
        [("target L", clean[crop]), ("L+n high", noisy["high"][crop])]
        + [(f"{method} high", outputs[(method, "high")][crop]) for method in METHODS],
        args.out_dir / "zooms_high.png",
        "High noise, center crop",
        columns=4,
    )

    per_image = ROOT / "results" / "classical_denoising" / "per_image.csv"
    worst = []
    if per_image.exists():
        with per_image.open(encoding="utf-8", newline="") as handle:
            rows = [
                row
                for row in csv.DictReader(handle)
                if row["dataset"] == "lolv2-real"
                and row["method"] == "bm3d"
                and row["noise_level"] == "high"
            ]
        worst = sorted(rows, key=lambda row: float(row["psnr"]))[:3]
    panels = []
    for row in worst:
        current = load_rgb(ROOT / row["image_id"])
        current_noisy = noise_config.add(current, "high", image_id=row["image_id"])
        current_bm3d = build_denoiser(
            "bm3d", frozen["methods"]["bm3d"]["high"]["params"]
        ).denoise(current_noisy)
        name = Path(row["image_id"]).name
        panels.extend(
            [
                (f"{name} target", current),
                (f"{name} noisy", current_noisy),
                (f"{name} bm3d {float(row['psnr']):.2f} dB", current_bm3d),
            ]
        )
    if panels:
        render(
            panels,
            args.out_dir / "failure_cases.png",
            "Worst BM3D cases, high noise (lolv2-real)",
            columns=3,
        )

    print(f"figures={args.out_dir}")
    print(f"example={image_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
