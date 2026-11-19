"""Create deterministic failure-analysis panels for Stage 07 (test results only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lowlight.metrics import PsnrMetric, resolve_device  # noqa: E402
from lowlight.pipelines import build_pipeline  # noqa: E402
from lowlight.utils.io import load_rgb, read_csv_rows  # noqa: E402
from lowlight.utils.noise import load_noise_config  # noqa: E402

PER_IMAGE = ROOT / "results" / "pipelines" / "evaluation" / "per_image.csv"
REAL_METHODS = {
    "Dc-Bc": ("configs/pipelines/classical_classical.yaml", None),
    "Bc-Dc": ("configs/pipelines/brighten_then_denoise.yaml", None),
    "Dcnn-Bcnn": ("configs/pipelines/cnn_cnn.yaml", None),
    "E2E real": ("configs/pipelines/end_to_end.yaml", "real"),
}
HIGH_METHODS = {
    "Dc-Bc": ("configs/pipelines/classical_classical.yaml", None),
    "Bc-Dc": ("configs/pipelines/brighten_then_denoise.yaml", None),
    "Dcnn-Bcnn": ("configs/pipelines/cnn_cnn.yaml", None),
    "E2E controlled": ("configs/pipelines/end_to_end.yaml", "controlled"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/ablations/failure_panels")
    parser.add_argument(
        "--handout-dir",
        type=Path,
        default=ROOT / "Outdated/Plans/handouts/stage07/figures",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def image_scores(
    rows: list[dict[str, str]], pipeline: str, level: str
) -> dict[tuple[str, str], float]:
    buckets: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        if row["pipeline"] != pipeline or row["noise_level"] != level:
            continue
        key = (row["dataset"], row["image_id"])
        buckets.setdefault(key, []).append(float(row["psnr"]))
    return {key: float(np.mean(values)) for key, values in buckets.items()}


def choose_cases(rows: list[dict[str, str]]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    real = image_scores(rows, "end_to_end_real", "real")
    ordered_real = sorted(real, key=lambda key: real[key])
    real_cases = [*ordered_real[:3], ordered_real[-1]]
    high = image_scores(rows, "brighten_then_denoise", "high")
    high_cases = sorted(high, key=lambda key: high[key])[:3]
    return real_cases, high_cases


def make_figure(
    cases: list[tuple[str, str]],
    *,
    level: str,
    methods: dict[str, tuple[str, str | None]],
    output: Path,
    device: object,
    noise_config: object,
    title: str,
) -> None:
    pipelines = {
        label: build_pipeline(config, root=ROOT, seed=0, device=device, variant=variant)  # type: ignore[arg-type]
        for label, (config, variant) in methods.items()
    }
    psnr = PsnrMetric()
    targets = {}
    for split in ("lol_test.csv", "lolv2_test.csv"):
        for row in read_csv_rows(ROOT / "data" / "splits" / split):
            targets[row["low_path"]] = row["gt_path"]
    n_rows = len(cases) * 2
    fig, axes = plt.subplots(n_rows, 6, figsize=(17.5, 3.2 * n_rows))
    for case_index, (dataset, image_id) in enumerate(cases):
        target = load_rgb(ROOT / targets[image_id]).astype(np.float64)
        low = load_rgb(ROOT / image_id)
        if level == "high":
            source = noise_config.add(low, "high", image_id=image_id, seed=0)  # type: ignore[attr-defined]
        else:
            source = low
        predictions = {
            label: np.asarray(pipeline.run(source), dtype=np.float64)
            for label, pipeline in pipelines.items()
        }
        rgb_row = case_index * 2
        error_row = rgb_row + 1
        panels = [("input", source), ("GT", target)]
        for label, prediction in predictions.items():
            panels.append((label, prediction))
        for column, (label, image) in enumerate(panels):
            axis = axes[rgb_row, column]
            axis.imshow(np.clip(image, 0.0, 1.0))
            axis.set_title(
                f"{dataset}: {Path(image_id).name}" if column == 0 else label,
                fontsize=9,
            )
            axis.axis("off")
            if label in predictions:
                score = float(psnr.compute(predictions[label], target))
                axis.text(
                    0.5,
                    -0.08,
                    f"PSNR {score:.2f} dB",
                    transform=axis.transAxes,
                    ha="center",
                    va="top",
                    fontsize=8,
                )
        for column in range(6):
            axes[error_row, column].axis("off")
        for offset, label in enumerate(predictions, start=2):
            error = np.abs(predictions[label] - target).mean(axis=2)
            axis = axes[error_row, offset]
            axis.imshow(error, cmap="inferno", vmin=0.0, vmax=0.35)
            axis.set_title("|prediction - GT|", fontsize=8)
            axis.axis("off")
        axes[error_row, 0].text(
            0.0,
            0.5,
            f"{dataset}\n{Path(image_id).name}",
            transform=axes[error_row, 0].transAxes,
            fontsize=8,
            va="center",
        )
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"wrote {output}")


def main() -> int:
    args = parse_args()
    rows = read_csv_rows(PER_IMAGE)
    real_cases, high_cases = choose_cases(rows)
    device = resolve_device(None if args.device == "auto" else args.device)
    noise_config = load_noise_config(ROOT / "configs/data/noise.yaml")
    make_figure(
        real_cases,
        level="real",
        methods=REAL_METHODS,
        output=args.out_dir / "failure_real.png",
        device=device,
        noise_config=noise_config,
        title="Real input: 3 worst + best E2E cases (seed 0, fixed rule)",
    )
    make_figure(
        high_cases,
        level="high",
        methods=HIGH_METHODS,
        output=args.out_dir / "failure_high.png",
        device=device,
        noise_config=noise_config,
        title="High noise: 3 worst Bc-Dc cases (seed 0, fixed rule)",
    )
    for name in ("failure_real.png", "failure_high.png"):
        source = args.out_dir / name
        target = args.handout_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
