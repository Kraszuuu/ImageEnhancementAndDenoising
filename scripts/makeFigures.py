"""Create Stage 07 summary figures from analysis tables."""

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

from lowlight.utils.io import read_csv_rows  # noqa: E402

LEVELS = ["clean", "low", "medium", "high"]
CONTROLLED = ["low", "medium", "high"]
LABELS = {
    "bm3d": "BM3D",
    "bilateral": "bilateral",
    "cnn_unet": "CNN U-Net",
    "gamma": "gamma",
    "classical_classical": "Dc-Bc",
    "brighten_then_denoise": "Bc-Dc",
    "classical_cnn": "Dc-Bcnn",
    "cnn_classical": "Dcnn-Bc",
    "cnn_cnn": "Dcnn-Bcnn",
    "end_to_end_real": "E2E real",
    "end_to_end_controlled": "E2E controlled",
    "l1_real": "E2E L1",
    "lol_only": "LOL only",
    "lolv2_only": "LOL-v2 only",
    "synthetic_brightening": "synthetic brightening",
    "synthetic_denoising": "synthetic denoising",
}


def rows(path: Path) -> list[dict[str, str]]:
    return read_csv_rows(path)


def find(
    table: list[dict[str, str]],
    *,
    method: str,
    metric: str,
    dataset: str | None = None,
    noise_level: str | None = None,
    track: str | None = None,
) -> dict[str, str] | None:
    for row in table:
        if row.get("method") != method or row.get("metric") != metric:
            continue
        if dataset is not None and row.get("dataset") != dataset:
            continue
        if noise_level is not None and row.get("noise_level") != noise_level:
            continue
        if track is not None and row.get("track") != track:
            continue
        return row
    return None


def value(row: dict[str, str] | None, column: str = "mean") -> float:
    if row is None:
        return float("nan")
    return float(row[column])


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_noise(summary: list[dict[str, str]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for axis, dataset in zip(axes, ("lol", "lolv2-real"), strict=True):
        for method in ("bm3d", "bilateral", "cnn_unet"):
            x = list(range(len(LEVELS)))
            y = [
                value(
                    find(
                        summary,
                        method=method,
                        metric="psnr",
                        dataset=dataset,
                        noise_level=level,
                    )
                )
                for level in LEVELS
            ]
            axis.plot(x, y, marker="o", label=LABELS[method])
        axis.set_xticks(range(len(LEVELS)), LEVELS)
        axis.set_title(dataset)
        axis.set_xlabel("noise level")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("PSNR [dB]")
    axes[0].legend(frameon=False)
    save(fig, output / "denoising_noise_levels.png")


def plot_brightening(summary: list[dict[str, str]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    methods = ("gamma", "cnn_unet")
    tracks = ("raw", "d_fixed")
    width = 0.36
    for axis, metric, ylabel in zip(axes, ("psnr", "lpips"), ("PSNR [dB]", "LPIPS"), strict=True):
        positions = np.arange(len(tracks))
        for offset, method in zip((-width / 2, width / 2), methods, strict=True):
            values = [
                value(
                    find(
                        summary,
                        method=method,
                        metric=metric,
                        track=track,
                        dataset="lolv2-real",
                    )
                )
                for track in tracks
            ]
            axis.bar(positions + offset, values, width, label=LABELS[method])
        axis.set_xticks(positions, tracks)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Brightening on LOL-v2-real")
    save(fig, output / "brightening_tracks.png")


def plot_pipeline_levels(summary: list[dict[str, str]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    real_methods = (
        "classical_classical",
        "brighten_then_denoise",
        "classical_cnn",
        "cnn_classical",
        "cnn_cnn",
        "end_to_end_real",
    )
    real_values = [
        value(find(summary, method=method, metric="psnr", dataset="lolv2-real", noise_level="real"))
        for method in real_methods
    ]
    axes[0].barh([LABELS[method] for method in real_methods], real_values, color="#4c72b0")
    axes[0].set_xlabel("PSNR [dB]")
    axes[0].set_title("real input")

    controlled_methods = (
        "classical_classical",
        "brighten_then_denoise",
        "classical_cnn",
        "cnn_cnn",
        "end_to_end_controlled",
    )
    for method in controlled_methods:
        values = [
            value(
                find(
                    summary,
                    method=method,
                    metric="psnr",
                    dataset="lolv2-real",
                    noise_level=level,
                )
            )
            for level in CONTROLLED
        ]
        axes[1].plot(CONTROLLED, values, marker="o", label=LABELS[method])
    axes[1].set_xlabel("controlled noise level")
    axes[1].set_ylabel("PSNR [dB]")
    axes[1].set_title("controlled noise")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, output / "pipeline_levels.png")


def plot_dataset_effect(summary: list[dict[str, str]], output: Path) -> None:
    methods = ("cnn_cnn", "end_to_end_real")
    datasets = ("lol", "lolv2-real")
    positions = np.arange(len(methods))
    width = 0.36
    fig, axis = plt.subplots(figsize=(6.6, 4.2))
    for offset, dataset in zip((-width / 2, width / 2), datasets, strict=True):
        values = [
            value(find(summary, method=method, metric="psnr", dataset=dataset, noise_level="real"))
            for method in methods
        ]
        axis.bar(positions + offset, values, width, label=dataset)
    axis.set_xticks(positions, [LABELS[method] for method in methods])
    axis.set_ylabel("PSNR [dB]")
    axis.set_title("Real pipeline: dataset effect")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    save(fig, output / "dataset_effect.png")


def plot_effects(tests: list[dict[str, str]], output: Path) -> None:
    psnr = [row for row in tests if row["metric"] == "psnr" and row["dataset"] == "lolv2-real"]
    labels = [f"{row['comparison']}\n(B-A)" for row in psnr]
    deltas = [-float(row["mean_delta"]) for row in psnr]
    adjusted = [float(row["wilcoxon_p_holm"]) for row in psnr]
    order = np.argsort(deltas)
    labels = [labels[index] for index in order]
    deltas = [deltas[index] for index in order]
    adjusted = [adjusted[index] for index in order]
    colors = ["#55a868" if delta >= 0.0 else "#c44e52" for delta in deltas]
    fig, axis = plt.subplots(figsize=(9.5, 5.6))
    axis.barh(labels, deltas, color=colors)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("PSNR difference, method B - method A [dB]")
    axis.set_title("Pre-registered PSNR comparisons")
    for index, p_value in enumerate(adjusted):
        if p_value < 0.001:
            stars = "***"
        elif p_value < 0.01:
            stars = "**"
        elif p_value < 0.05:
            stars = "*"
        else:
            stars = "n.s."
        axis.text(deltas[index], index, f" {stars}", va="center", fontsize=8)
    axis.grid(axis="x", alpha=0.25)
    save(fig, output / "paired_psnr_effects.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "results/ablations")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/ablations/figures")
    args = parser.parse_args()
    analysis_dir = args.analysis_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_noise(rows(analysis_dir / "summary_denoising_noise.csv"), output_dir)
    plot_brightening(rows(analysis_dir / "summary_brightening_tracks.csv"), output_dir)
    plot_pipeline_levels(rows(analysis_dir / "summary_pipeline_levels.csv"), output_dir)
    plot_dataset_effect(rows(analysis_dir / "summary_pipeline_levels.csv"), output_dir)
    plot_effects(rows(analysis_dir / "paired_tests.csv"), output_dir)
    plot_composition(
        rows(analysis_dir / "summary_composition_levels.csv"),
        rows(analysis_dir / "summary_pipeline_levels.csv"),
        output_dir,
    )
    plot_synthetic(rows(analysis_dir / "summary_synthetic_control.csv"), output_dir)
    plot_loss(
        rows(analysis_dir / "summary_loss_test.csv"),
        rows(analysis_dir / "summary_pipeline_levels.csv"),
        output_dir,
    )
    print(f"wrote figures to {output_dir}")


def plot_composition(
    composition: list[dict[str, str]],
    pipeline: list[dict[str, str]],
    output: Path,
) -> None:
    datasets = ("lol", "lolv2-real")
    methods = ("lol_only", "lolv2_only", "end_to_end_real")
    colors = ("#4c72b0", "#dd8452", "#55a868")
    positions = np.arange(len(methods))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for axis, dataset in zip(axes, datasets, strict=True):
        values = []
        for method in methods:
            source = pipeline if method == "end_to_end_real" else composition
            values.append(value(find(source, method=method, metric="psnr", dataset=dataset)))
        axis.bar(positions, values, color=colors)
        axis.set_xticks(positions, [LABELS[method] for method in methods], rotation=12)
        axis.set_title(f"test dataset: {dataset}")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("PSNR [dB]")
    fig.suptitle("Dataset-composition ablation (real input)")
    save(fig, output / "composition_effect.png")


def plot_synthetic(summary: list[dict[str, str]], output: Path) -> None:
    datasets = ("lol", "lolv2-real")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    positions = np.arange(len(datasets))
    values = [
        value(
            find(
                summary,
                method="synthetic_brightening",
                metric="psnr",
                dataset=dataset,
                noise_level="synthetic",
            )
        )
        for dataset in datasets
    ]
    errors = [
        value(
            find(
                summary,
                method="synthetic_brightening",
                metric="psnr",
                dataset=dataset,
                noise_level="synthetic",
            ),
            "seed_std",
        )
        for dataset in datasets
    ]
    axes[0].bar(positions, values, yerr=errors, capsize=4, color="#4c72b0")
    axes[0].set_xticks(positions, datasets)
    axes[0].set_ylabel("PSNR [dB]")
    axes[0].set_title("darken(G) -> G")
    axes[0].grid(axis="y", alpha=0.25)

    for dataset in datasets:
        series = [
            value(
                find(
                    summary,
                    method="synthetic_denoising",
                    metric="psnr",
                    dataset=dataset,
                    noise_level=level,
                )
            )
            for level in CONTROLLED
        ]
        errors = [
            value(
                find(
                    summary,
                    method="synthetic_denoising",
                    metric="psnr",
                    dataset=dataset,
                    noise_level=level,
                ),
                "seed_std",
            )
            for level in CONTROLLED
        ]
        axes[1].errorbar(CONTROLLED, series, yerr=errors, marker="o", capsize=4, label=dataset)
    axes[1].set_xlabel("noise level")
    axes[1].set_ylabel("PSNR [dB]")
    axes[1].set_title("darken(G)+n -> darken(G)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.suptitle("Synthetic-control ablation (error bars: between-seed SD)")
    save(fig, output / "synthetic_control.png")


def plot_loss(
    loss_summary: list[dict[str, str]],
    pipeline: list[dict[str, str]],
    output: Path,
) -> None:
    datasets = ("lol", "lolv2-real")
    positions = np.arange(len(datasets))
    width = 0.36
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    for offset, method, source in zip(
        (-width / 2, width / 2),
        ("l1_real", "end_to_end_real"),
        (loss_summary, pipeline),
        strict=True,
    ):
        values = [
            value(find(source, method=method, metric="psnr", dataset=dataset))
            for dataset in datasets
        ]
        errors = [
            value(find(source, method=method, metric="psnr", dataset=dataset), "seed_std")
            for dataset in datasets
        ]
        axis.bar(positions + offset, values, width, yerr=errors, capsize=4, label=LABELS[method])
    axis.set_xticks(positions, datasets)
    axis.set_ylabel("PSNR [dB]")
    axis.set_title("Loss-function ablation (error bars: between-seed SD)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    save(fig, output / "loss_test.png")


if __name__ == "__main__":
    main()
