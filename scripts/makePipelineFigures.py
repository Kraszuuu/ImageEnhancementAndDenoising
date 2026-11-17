"""Create Stage 06 pipeline comparison figures and failure panels."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lowlight.metrics import PsnrMetric  # noqa: E402
from lowlight.pipelines import build_pipeline  # noqa: E402
from lowlight.utils.io import load_rgb, read_csv_rows  # noqa: E402
from lowlight.utils.noise import load_noise_config  # noqa: E402

DISPLAY_GAMMA = 0.45
SPLITS = ("lol_test.csv", "lolv2_test.csv")
PIPELINES = {
    "classical_classical": ("configs/pipelines/classical_classical.yaml", None),
    "brighten_then_denoise": ("configs/pipelines/brighten_then_denoise.yaml", None),
    "classical_cnn": ("configs/pipelines/classical_cnn.yaml", None),
    "cnn_classical": ("configs/pipelines/cnn_classical.yaml", None),
    "cnn_cnn": ("configs/pipelines/cnn_cnn.yaml", None),
    "end_to_end_real": ("configs/pipelines/end_to_end.yaml", "real"),
    "end_to_end_controlled": ("configs/pipelines/end_to_end.yaml", "controlled"),
}
SHORT = {
    "classical_classical": "Dc-Bc",
    "brighten_then_denoise": "Bc-Dc",
    "classical_cnn": "Dc-Bcnn",
    "cnn_classical": "Dcnn-Bc",
    "cnn_cnn": "Dcnn-Bcnn",
    "end_to_end_real": "E2E real",
    "end_to_end_controlled": "E2E controlled",
}
REAL_PIPELINES = (
    "classical_classical",
    "brighten_then_denoise",
    "classical_cnn",
    "cnn_classical",
    "cnn_cnn",
    "end_to_end_real",
)
CONTROLLED_PIPELINES = (
    "classical_classical",
    "brighten_then_denoise",
    "classical_cnn",
    "cnn_classical",
    "cnn_cnn",
    "end_to_end_controlled",
)


def display(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0.0, 1.0) ** DISPLAY_GAMMA


def load_test_rows() -> list[dict[str, str]]:
    rows = []
    for split in SPLITS:
        rows.extend(read_csv_rows(ROOT / "data" / "splits" / split))
    return rows


def load_pipeline(name: str, device: torch.device, seed: int = 0):
    config, variant = PIPELINES[name]
    return build_pipeline(
        ROOT / config,
        seed=seed,
        device=device,
        variant=variant,
    )


def per_image_seed_means(
    rows: list[dict[str, str]], pipeline: str, level: str
) -> dict[tuple[str, str], float]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["pipeline"] == pipeline and row["noise_level"] == level:
            grouped[(row["dataset"], row["image_id"])].append(float(row["psnr"]))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def pooled_psnr(rows: list[dict[str, str]], pipeline: str, level: str) -> float:
    values = per_image_seed_means(rows, pipeline, level)
    return float(np.mean(list(values.values())))


def mean_time(rows: list[dict[str, str]], pipeline: str, level: str) -> float:
    values = [
        float(row["total_ms"])
        for row in rows
        if row["pipeline"] == pipeline and row["noise_level"] == level
    ]
    return float(np.mean(values))


def validation_curves(out_dir: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7))
    panels = (
        ("Pipeline brightener: D_classical", "brighteners/classical", axes[0, 0]),
        ("Pipeline brightener: D_cnn", "brighteners/cnn", axes[0, 1]),
        ("End-to-end: real", "end_to_end/real", axes[1, 0]),
        ("End-to-end: controlled", "end_to_end/controlled", axes[1, 1]),
    )
    for title, relative, axis in panels:
        for seed in (0, 1, 2):
            path = ROOT / "results" / "pipelines" / relative / f"seed{seed}" / "history.csv"
            rows = read_csv_rows(path)
            axis.plot(
                [int(row["epoch"]) for row in rows],
                [float(row["val_psnr"]) for row in rows],
                label=f"seed {seed}",
            )
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("epoch")
        axis.set_ylabel("val PSNR [dB]")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Stage 06 validation PSNR", fontsize=12)
    figure.tight_layout()
    figure.savefig(out_dir / "validation_curves.png", dpi=150)
    plt.close(figure)


def quality_by_level(evaluation_rows: list[dict[str, str]], out_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    panels = (
        ("Real input", "real", REAL_PIPELINES, axes[0]),
        ("Controlled noise", None, CONTROLLED_PIPELINES, axes[1]),
    )
    for title, fixed_level, pipelines, axis in panels:
        levels = ["real"] if fixed_level else ["low", "medium", "high"]
        width = 0.8 / len(pipelines)
        positions = np.arange(len(levels))
        for index, pipeline in enumerate(pipelines):
            scores = [pooled_psnr(evaluation_rows, pipeline, level) for level in levels]
            axis.bar(
                positions + (index - (len(pipelines) - 1) / 2) * width,
                scores,
                width,
                label=SHORT[pipeline],
            )
        axis.set_xticks(positions, levels)
        axis.set_title(title, fontsize=10)
        axis.set_ylabel("PSNR [dB]")
        axis.set_ylim(8, 24)
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
    figure.suptitle("Stage 06 PSNR by pipeline and noise level, pooled test", fontsize=12)
    figure.tight_layout()
    figure.savefig(out_dir / "psnr_by_level.png", dpi=150)
    plt.close(figure)


def quality_vs_time(evaluation_rows: list[dict[str, str]], out_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    panels = (
        ("Real input", "real", REAL_PIPELINES, axes[0]),
        ("Controlled high", "high", CONTROLLED_PIPELINES, axes[1]),
    )
    for title, level, pipelines, axis in panels:
        for pipeline in pipelines:
            x = mean_time(evaluation_rows, pipeline, level)
            y = pooled_psnr(evaluation_rows, pipeline, level)
            axis.scatter(x, y, s=45)
            axis.annotate(
                SHORT[pipeline], (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8
            )
        axis.set_xscale("log")
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("mean total time [ms, log scale]")
        axis.set_ylabel("PSNR [dB]")
        axis.grid(alpha=0.25)
    figure.suptitle("Stage 06 quality vs latency, pooled test", fontsize=12)
    figure.tight_layout()
    figure.savefig(out_dir / "quality_vs_time.png", dpi=150)
    plt.close(figure)


def median_brightness_row() -> dict[str, str]:
    rows = load_test_rows()
    scored = sorted(
        (float(load_rgb(ROOT / row["low_path"]).mean()), row)
        for row in rows
        if row["dataset"] == "lolv2-real"
    )
    return scored[len(scored) // 2][1]


def render_panels(
    panels: list[tuple[str, np.ndarray]],
    titles: list[str],
    path: Path,
    title: str,
    columns: int,
) -> None:
    count = len(panels)
    rows = int(np.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 2.9 * rows), squeeze=False)
    for axis in axes.ravel():
        axis.axis("off")
    for axis, (label, image), subtitle in zip(axes.ravel(), panels, titles, strict=False):
        axis.imshow(display(image))
        axis.set_title(f"{label}\n{subtitle}", fontsize=8)
        axis.axis("off")
    figure.suptitle(title, fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def qualitative(
    out_dir: Path,
    device: torch.device,
    noise_config,
) -> None:
    row = median_brightness_row()
    clean = load_rgb(ROOT / row["gt_path"])
    low = load_rgb(ROOT / row["low_path"])
    high = noise_config.add(low, "high", image_id=row["low_path"], seed=0)
    metric = PsnrMetric()
    real_outputs = {
        "L": low,
        "G": clean,
        "Dc-Bc": load_pipeline("classical_classical", device).run(low),
        "Dc-Bcnn": load_pipeline("classical_cnn", device).run(low),
        "Dcnn-Bcnn": load_pipeline("cnn_cnn", device).run(low),
        "E2E real": load_pipeline("end_to_end_real", device).run(low),
    }
    panels = list(real_outputs.items())
    scores = [
        "" if label in {"G"} else f"PSNR {metric.compute(image, clean):.2f} dB"
        for label, image in panels
    ]
    render_panels(
        panels,
        scores,
        out_dir / "real_examples.png",
        f"Stage 06 real input, LOL-v2 Real {Path(row['low_path']).stem}",
        columns=3,
    )

    controlled_outputs = {
        "L+n high": high,
        "G": clean,
        "Bc-Dc": load_pipeline("brighten_then_denoise", device).run(high),
        "Dc-Bc": load_pipeline("classical_classical", device).run(high),
        "Dc-Bcnn": load_pipeline("classical_cnn", device).run(high),
        "Dcnn-Bcnn": load_pipeline("cnn_cnn", device).run(high),
        "E2E controlled": load_pipeline("end_to_end_controlled", device).run(high),
    }
    panels = list(controlled_outputs.items())
    scores = [
        "" if label in {"G"} else f"PSNR {metric.compute(image, clean):.2f} dB"
        for label, image in panels
    ]
    render_panels(
        panels,
        scores,
        out_dir / "controlled_high_examples.png",
        f"Stage 06 controlled high noise, LOL-v2 Real {Path(row['low_path']).stem}",
        columns=4,
    )


def worst_rows(
    evaluation_rows: list[dict[str, str]], pipeline: str, level: str
) -> list[tuple[str, str, float]]:
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    scores = per_image_seed_means(evaluation_rows, pipeline, level)
    for (dataset, image_id), score in scores.items():
        grouped[dataset].append((image_id, score))
    selected = []
    for dataset in sorted(grouped):
        image_id, score = min(grouped[dataset], key=lambda item: item[1])
        selected.append((dataset, image_id, score))
    return selected


def failures(
    out_dir: Path,
    device: torch.device,
    noise_config,
    evaluation_rows: list[dict[str, str]],
) -> None:
    metric = PsnrMetric()
    panels: list[tuple[str, np.ndarray]] = []
    titles: list[str] = []
    pipelines = {
        "Dcnn-Bcnn": load_pipeline("cnn_cnn", device),
        "E2E real": load_pipeline("end_to_end_real", device),
        "E2E controlled": load_pipeline("end_to_end_controlled", device),
    }
    cases = [
        ("real", "end_to_end_real", "E2E real"),
        ("high", "end_to_end_controlled", "E2E controlled"),
    ]
    for level, pipeline, best in cases:
        for dataset, image_id, score in worst_rows(evaluation_rows, pipeline, level):
            split_row = next(row for row in load_test_rows() if row["low_path"] == image_id)
            clean = load_rgb(ROOT / split_row["gt_path"])
            low = load_rgb(ROOT / split_row["low_path"])
            image = (
                low if level == "real" else noise_config.add(low, "high", image_id=image_id, seed=0)
            )
            panels.extend(
                [
                    (
                        f"{dataset} {Path(image_id).stem}\nL"
                        if level == "real"
                        else f"{dataset} {Path(image_id).stem}\nL+n high",
                        image,
                    ),
                    (f"{dataset} {Path(image_id).stem}\nG", clean),
                    ("Dcnn-Bcnn", pipelines["Dcnn-Bcnn"].run(image)),
                    (best, pipelines[best].run(image)),
                ]
            )
            titles.extend(
                [
                    "",
                    "",
                    f"PSNR {metric.compute(panels[-2][1], clean):.2f} dB",
                    f"PSNR {score:.2f} dB",
                ]
            )
    render_panels(
        panels,
        titles,
        out_dir / "failure_cases.png",
        "Stage 06 failure cases: worst E2E per dataset",
        columns=4,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "pipelines" / "figures")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=ROOT / "results" / "pipelines" / "evaluation" / "per_image.csv",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    evaluation_rows = read_csv_rows(args.evaluation)
    noise_config = load_noise_config(ROOT / "configs" / "data" / "noise.yaml")
    validation_curves(args.out_dir)
    quality_by_level(evaluation_rows, args.out_dir)
    quality_vs_time(evaluation_rows, args.out_dir)
    qualitative(args.out_dir, device, noise_config)
    failures(args.out_dir, device, noise_config, evaluation_rows)
    print(f"figures={args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
