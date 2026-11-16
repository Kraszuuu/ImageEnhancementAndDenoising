"""Evaluate frozen Stage 05 CNN brighteners on the locked test splits (run once)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lowlight.classical.brighteners import build_brightener  # noqa: E402
from lowlight.classical.denoisers import build_denoiser  # noqa: E402
from lowlight.metrics import (  # noqa: E402
    LpipsMetric,
    NiqeMetric,
    PsnrMetric,
    SsimMetric,
    resolve_device,
)
from lowlight.models import build_model, count_parameters, denoise_full_image  # noqa: E402
from lowlight.training.checkpointing import CheckpointManager  # noqa: E402
from lowlight.utils.io import load_rgb, read_csv_rows, write_csv_rows  # noqa: E402

TEST_SPLITS = ("lol_test.csv", "lolv2_test.csv")
METRIC_NAMES = ("psnr", "ssim", "lpips", "niqe")
PER_IMAGE_COLUMNS = [
    "run_id",
    "dataset",
    "split",
    "track",
    "method",
    "seed",
    "image_id",
    "psnr",
    "ssim",
    "lpips",
    "niqe",
    "brightness_error",
    "color_cast",
    "clip_low_fraction",
    "clip_high_fraction",
    "inference_ms",
    "source_ms",
    "width",
    "height",
]
SUMMARY_COLUMNS = [
    "track",
    "method",
    "seed",
    "dataset",
    "metric",
    "mean",
    "std",
    "median",
    "ci95_low",
    "ci95_high",
    "n",
]
SEED_SUMMARY_COLUMNS = [
    "track",
    "method",
    "dataset",
    "metric",
    "seed_mean",
    "seed_std",
    "n_seeds",
]
COMPARISON_COLUMNS = [
    "track",
    "dataset",
    "metric",
    "method",
    "mean",
    "std",
    "median",
    "ci95_low",
    "ci95_high",
    "n",
]
TEST_COLUMNS = [
    "track",
    "metric",
    "baseline",
    "n_pairs",
    "cnn_mean",
    "baseline_mean",
    "mean_delta",
    "median_delta",
    "wilcoxon_stat",
    "wilcoxon_p",
    "ttest_stat",
    "ttest_p",
]
TIMING_COLUMNS = [
    "track",
    "method",
    "dataset",
    "device",
    "seed",
    "mean_ms",
    "std_ms",
    "median_ms",
    "n",
    "protocol",
    "source_median_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT / "checkpoints" / "brightening_cnn",
    )
    parser.add_argument("--tracks", nargs="+", default=["raw", "d_fixed"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--splits", nargs="*", default=list(TEST_SPLITS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "brightening_cnn" / "evaluation",
    )
    parser.add_argument(
        "--classical-per-image",
        type=Path,
        default=ROOT / "results" / "classical_brightening" / "per_image.csv",
    )
    parser.add_argument(
        "--frozen",
        type=Path,
        default=ROOT / "configs" / "classical" / "brightening_frozen.yaml",
    )
    parser.add_argument(
        "--d-fixed",
        type=Path,
        default=ROOT / "configs" / "pipelines" / "d_fixed.yaml",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--timing-iters", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def format_cell(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.6f}"
    return value


def load_rows(splits: list[str], limit: int) -> list[dict[str, str]]:
    rows = []
    for name in splits:
        rows.extend(read_csv_rows(ROOT / "data" / "splits" / name))
    if limit > 0:
        rows = rows[:limit]
    return rows


def bootstrap_ci(
    values: np.ndarray, resamples: int = 1000, ci: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    low, high = np.percentile(means, [alpha * 100.0, (1.0 - alpha) * 100.0])
    return float(low), float(high)


def summarize_values(values: np.ndarray) -> dict[str, object]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "mean": math.nan,
            "std": math.nan,
            "median": math.nan,
            "ci95_low": math.nan,
            "ci95_high": math.nan,
            "n": 0,
        }
    low, high = bootstrap_ci(finite)
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "median": float(np.median(finite)),
        "ci95_low": low,
        "ci95_high": high,
        "n": int(finite.size),
    }


def summarize(
    rows: list[dict[str, object]],
    *,
    methods: list[str],
    tracks: list[str],
    seeds: list[int | str],
) -> list[dict[str, object]]:
    output = []
    datasets = sorted({str(row["dataset"]) for row in rows})
    for track in tracks:
        for method in methods:
            for seed in seeds:
                seed_rows = [
                    row
                    for row in rows
                    if row["track"] == track
                    and row["method"] == method
                    and (seed == "all" or row["seed"] == seed)
                ]
                for dataset in [*datasets, "all"]:
                    subset = (
                        seed_rows
                        if dataset == "all"
                        else [row for row in seed_rows if row["dataset"] == dataset]
                    )
                    if not subset:
                        continue
                    for metric in METRIC_NAMES:
                        values = np.asarray(
                            [float(row[metric]) for row in subset], dtype=np.float64
                        )
                        scores = summarize_values(values)
                        output.append(
                            {
                                "track": track,
                                "method": method,
                                "seed": seed,
                                "dataset": dataset,
                                "metric": metric,
                                **scores,
                            }
                        )
    return output


def seed_summary(
    summary_rows: list[dict[str, object]],
    *,
    methods: list[str],
    tracks: list[str],
) -> list[dict[str, object]]:
    output = []
    for track in tracks:
        for method in methods:
            for metric in METRIC_NAMES:
                per_seed = [
                    float(row["mean"])
                    for row in summary_rows
                    if row["track"] == track
                    and row["method"] == method
                    and row["dataset"] == "all"
                    and row["metric"] == metric
                    and row["seed"] != "all"
                ]
                if not per_seed:
                    continue
                values = np.asarray(per_seed, dtype=np.float64)
                output.append(
                    {
                        "track": track,
                        "method": method,
                        "dataset": "all",
                        "metric": metric,
                        "seed_mean": float(values.mean()),
                        "seed_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                        "n_seeds": int(values.size),
                    }
                )
    return output


def paired_tests(
    cnn_rows: list[dict[str, object]],
    baseline_rows: list[dict[str, object]],
    *,
    track: str,
    baseline: str,
) -> list[dict[str, object]]:
    try:
        from scipy.stats import ttest_rel, wilcoxon
    except ImportError:
        return []

    output = []
    by_baseline = {str(row["image_id"]): row for row in baseline_rows}
    for metric in METRIC_NAMES:
        cnn_by_image: dict[str, list[float]] = {}
        for row in cnn_rows:
            if row["track"] != track:
                continue
            cnn_by_image.setdefault(str(row["image_id"]), []).append(float(row[metric]))
        pairs = []
        for image_id, values in cnn_by_image.items():
            if image_id not in by_baseline:
                continue
            baseline_value = float(by_baseline[image_id][metric])
            cnn_value = float(np.mean(values))
            if math.isfinite(cnn_value) and math.isfinite(baseline_value):
                pairs.append((cnn_value, baseline_value))
        if len(pairs) < 2:
            continue
        cnn_values = np.asarray([pair[0] for pair in pairs])
        baseline_values = np.asarray([pair[1] for pair in pairs])
        differences = cnn_values - baseline_values
        try:
            wilcoxon_result = wilcoxon(differences)
            wilcoxon_stat = float(wilcoxon_result.statistic)
            wilcoxon_p = float(wilcoxon_result.pvalue)
        except ValueError:
            wilcoxon_stat = math.nan
            wilcoxon_p = math.nan
        ttest_result = ttest_rel(cnn_values, baseline_values)
        output.append(
            {
                "track": track,
                "metric": metric,
                "baseline": baseline,
                "n_pairs": len(pairs),
                "cnn_mean": float(cnn_values.mean()),
                "baseline_mean": float(baseline_values.mean()),
                "mean_delta": float(differences.mean()),
                "median_delta": float(np.median(differences)),
                "wilcoxon_stat": wilcoxon_stat,
                "wilcoxon_p": wilcoxon_p,
                "ttest_stat": float(ttest_result.statistic),
                "ttest_p": float(ttest_result.pvalue),
            }
        )
    return output


def metric_values(
    metrics: dict[str, object], prediction: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    return {
        name: float(metric.compute(prediction, target))  # type: ignore[attr-defined]
        for name, metric in metrics.items()
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def batch_lpips(
    metric: LpipsMetric, predictions: list[np.ndarray], targets: list[np.ndarray], batch: int = 8
) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(predictions), batch):
        pred = np.stack(
            [np.transpose(item, (2, 0, 1)) for item in predictions[start : start + batch]]
        ).astype(np.float32)
        ref = np.stack(
            [np.transpose(item, (2, 0, 1)) for item in targets[start : start + batch]]
        ).astype(np.float32)
        pred_t = torch.from_numpy(pred).to(metric.device) * 2.0 - 1.0
        ref_t = torch.from_numpy(ref).to(metric.device) * 2.0 - 1.0
        with (
            torch.inference_mode(),
            torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True),
        ):
            values = metric.model(pred_t, ref_t).reshape(-1).cpu().tolist()
        scores.extend(float(value) for value in values)
    return scores


def batch_niqe(metric: NiqeMetric, predictions: list[np.ndarray], batch: int = 8) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(predictions), batch):
        array = np.stack(
            [np.transpose(item, (2, 0, 1)) for item in predictions[start : start + batch]]
        ).astype(np.float32)
        tensor = torch.from_numpy(array).to(metric.device)
        with torch.inference_mode():
            values = metric.model(tensor).reshape(-1).cpu().tolist()
        scores.extend(float(value) for value in values)
    return scores


def color_stats(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred_mean = prediction.mean(axis=(0, 1))
    target_mean = target.mean(axis=(0, 1))
    return {
        "brightness_error": float(abs(prediction.mean() - target.mean())),
        "color_cast": float(np.max(np.abs(pred_mean - target_mean))),
        "clip_low_fraction": float(np.mean(prediction <= 0.01)),
        "clip_high_fraction": float(np.mean(prediction >= 0.99)),
    }


def predict_batch(
    model: torch.nn.Module, sources: list[np.ndarray], device: torch.device
) -> tuple[list[np.ndarray], float]:
    batch = np.stack(sources).astype(np.float32)
    tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(device)
    synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        output = denoise_full_image(model, tensor)
    synchronize(device)
    elapsed = (time.perf_counter() - started) * 1000.0 / len(sources)
    images = output.permute(0, 2, 3, 1).float().clamp(0.0, 1.0).cpu().numpy()
    return [image for image in images], elapsed


def timing_for_model(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        for _ in range(warmup):
            denoise_full_image(model, tensor)
        synchronize(device)
        samples = []
        for _ in range(iterations):
            synchronize(device)
            started = time.perf_counter()
            denoise_full_image(model, tensor)
            synchronize(device)
            samples.append((time.perf_counter() - started) * 1000.0)
    array = np.asarray(samples)
    return {
        "mean_ms": float(array.mean()),
        "std_ms": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median_ms": float(np.median(array)),
        "n": int(array.size),
    }


def save_validation_curves(path: Path, root: Path, tracks: list[str], seeds: list[int]) -> None:
    import csv

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for track in tracks:
        for seed in seeds:
            history_path = root / track / f"seed{seed}" / "history.csv"
            if not history_path.exists():
                continue
            with history_path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            epochs = [int(row["epoch"]) for row in rows]
            for axis, column in zip(axes, ["val_psnr", "val_ssim"], strict=True):
                axis.plot(epochs, [float(row[column]) for row in rows], label=f"{track} s{seed}")
    for axis, title in zip(axes, ["Validation PSNR", "Validation SSIM"], strict=True):
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_examples(
    path: Path,
    examples: list[dict[str, object]],
    gamma: object,
    seed: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = [
        "L",
        "D_fixed(L)",
        "target",
        f"CNN raw s{seed}",
        "gamma raw",
        f"CNN D_fixed s{seed}",
        "gamma D_fixed",
    ]
    figure, axes = plt.subplots(
        len(examples), len(columns), figsize=(3.0 * len(columns), 2.4 * len(examples))
    )
    axes = np.atleast_2d(axes)
    for row_index, example in enumerate(examples):
        low = np.asarray(example["low"])
        fixed = np.asarray(example["fixed"])
        target = np.asarray(example["target"])
        panels = [
            low,
            fixed,
            target,
            example["raw_cnn"],
            gamma.enhance(low),  # type: ignore[attr-defined]
            example["fixed_cnn"],
            gamma.enhance(fixed),  # type: ignore[attr-defined]
        ]
        for column_index, (label, image) in enumerate(zip(columns, panels, strict=True)):
            axis = axes[row_index, column_index]
            axis.imshow(np.clip(image, 0.0, 1.0))
            axis.set_title(f"{example['dataset']} {label}", fontsize=7)
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def save_detail(
    path: Path,
    example: dict[str, object],
    gamma: object,
    seed: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = ["L", "D_fixed(L)", "target", f"CNN D_fixed s{seed}", "gamma D_fixed"]
    low = np.asarray(example["low"])
    fixed = np.asarray(example["fixed"])
    panels = [low, fixed, example["target"], example["fixed_cnn"], gamma.enhance(fixed)]
    height, width = np.asarray(example["target"]).shape[:2]
    y0, y1 = height // 3, 2 * height // 3
    x0, x1 = width // 4, 3 * width // 4
    figure, axes = plt.subplots(2, len(columns), figsize=(3.0 * len(columns), 5.2))
    for row_index, (row_label, crop) in enumerate([("full", False), ("crop", True)]):
        for column_index, (label, image) in enumerate(zip(columns, panels, strict=True)):
            panel = np.asarray(image)
            if crop:
                panel = panel[y0:y1, x0:x1]
            axis = axes[row_index, column_index]
            axis.imshow(np.clip(panel, 0.0, 1.0))
            axis.set_title(f"{row_label} {label}", fontsize=7)
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_failures(
    path: Path,
    failures: list[dict[str, object]],
    gamma: object,
    seed: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = ["L", "D_fixed(L)", "target", f"CNN D_fixed s{seed}", "gamma D_fixed"]
    figure, axes = plt.subplots(
        len(failures), len(columns), figsize=(3.0 * len(columns), 2.4 * len(failures))
    )
    axes = np.atleast_2d(axes)
    for row_index, failure in enumerate(failures):
        low = np.asarray(failure["low"])
        fixed = np.asarray(failure["fixed"])
        panels = [
            low,
            fixed,
            failure["target"],
            failure["fixed_cnn"],
            gamma.enhance(fixed),  # type: ignore[attr-defined]
        ]
        for column_index, (label, image) in enumerate(zip(columns, panels, strict=True)):
            axis = axes[row_index, column_index]
            axis.imshow(np.clip(image, 0.0, 1.0))
            axis.set_title(f"{failure['dataset']} {label}", fontsize=7)
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    per_image_path = out_dir / "per_image.csv"
    if args.limit > 0 and tuple(args.splits) == TEST_SPLITS:
        raise SystemExit("--limit is forbidden on locked test splits")
    is_test = tuple(args.splits) == TEST_SPLITS and args.limit == 0
    if per_image_path.exists() and not args.force:
        raise SystemExit(
            f"refusing to overwrite {per_image_path}; test is locked "
            "(use --force only with a documented reason)"
        )
    tracks = [str(track) for track in args.tracks]
    seeds = [int(seed) for seed in args.seeds]
    checkpoints = {}
    for track in tracks:
        for seed in seeds:
            path = args.checkpoint_root / track / f"seed{seed}" / "best.pt"
            if not path.exists():
                raise SystemExit(f"missing checkpoint: {path}")
            checkpoints[(track, seed)] = path

    rows = load_rows(args.splits, args.limit)
    device = resolve_device(None if args.device == "auto" else args.device)
    metrics = {
        "psnr": PsnrMetric(),
        "ssim": SsimMetric(),
        "lpips": LpipsMetric(device=device),
        "niqe": NiqeMetric(device=device),
    }
    images = {row["low_path"]: load_rgb(ROOT / row["low_path"]) for row in rows}
    targets = {row["low_path"]: load_rgb(ROOT / row["gt_path"]) for row in rows}

    d_fixed_payload = yaml.safe_load(args.d_fixed.read_text(encoding="utf-8"))
    denoiser = build_denoiser(str(d_fixed_payload["name"]), dict(d_fixed_payload["params"]))
    print(f"images={len(rows)} tracks={tracks} seeds={seeds} device={device}")
    fixed_images: dict[str, np.ndarray] = {}
    fixed_ms: dict[str, float] = {}
    for index, row in enumerate(rows, start=1):
        low = images[row["low_path"]]
        started = time.perf_counter()
        fixed_images[row["low_path"]] = denoiser.denoise(low)
        fixed_ms[row["low_path"]] = (time.perf_counter() - started) * 1000.0
        if index % 10 == 0 or index == len(rows):
            print(f"  D_fixed {index}/{len(rows)}")

    sources = {"raw": images, "d_fixed": fixed_images}
    source_ms = {"raw": {image_id: 0.0 for image_id in images}, "d_fixed": fixed_ms}
    result_rows: list[dict[str, object]] = []
    predictions: dict[tuple[str, int, str], np.ndarray] = {}
    checkpoint_meta = []
    model_timing: dict[tuple[str, int, str], dict[str, object]] = {}
    for track in tracks:
        for seed in seeds:
            checkpoint_path = checkpoints[(track, seed)]
            payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
            checkpoint_track = str(payload.get("config", {}).get("track", ""))
            if checkpoint_track != track:
                raise SystemExit(f"checkpoint track mismatch: {checkpoint_path}")
            model_config = payload["config"]["model"]
            model = build_model(model_config).to(device).eval()
            CheckpointManager.load(checkpoint_path, model, device=device)
            model.eval()
            checkpoint_meta.append(
                {
                    "path": str(checkpoint_path.relative_to(ROOT)),
                    "sha256": sha256(checkpoint_path),
                    "track": track,
                    "seed": seed,
                    "epoch": payload.get("epoch"),
                    "val_psnr": payload.get("score"),
                    "parameters": count_parameters(model),
                }
            )
            print(
                f"track={track} seed={seed} epoch={payload.get('epoch')} "
                f"val_psnr={float(payload.get('score', math.nan)):.3f} "
                f"params={count_parameters(model)}"
            )
            seed_rows: list[dict[str, object]] = []
            seed_predictions: list[np.ndarray] = []
            seed_targets: list[np.ndarray] = []
            tasks = [row for row in rows]
            for start in range(0, len(tasks), args.batch_size):
                batch_rows = tasks[start : start + args.batch_size]
                batch_sources = [sources[track][row["low_path"]] for row in batch_rows]
                batch_predictions, elapsed = predict_batch(model, batch_sources, device)
                for row, prediction in zip(batch_rows, batch_predictions, strict=True):
                    image_id = row["low_path"]
                    target = targets[image_id]
                    scores = metric_values(metrics, prediction, target)
                    extra = color_stats(prediction, target)
                    height, width = target.shape[:2]
                    seed_rows.append(
                        {
                            "run_id": "stage05-cnn-brightening",
                            "dataset": row["dataset"],
                            "split": row["split"],
                            "track": track,
                            "method": "cnn_unet",
                            "seed": seed,
                            "image_id": image_id,
                            "psnr": scores["psnr"],
                            "ssim": scores["ssim"],
                            "lpips": 0.0,
                            "niqe": 0.0,
                            **extra,
                            "inference_ms": elapsed,
                            "source_ms": source_ms[track][image_id],
                            "width": width,
                            "height": height,
                        }
                    )
                    predictions[(track, seed, image_id)] = prediction
                    seed_predictions.append(prediction)
                    seed_targets.append(target)
                if start % 40 == 0:
                    print(
                        f"  {track} s{seed}: predictions "
                        f"{min(start + args.batch_size, len(tasks))}/{len(tasks)}"
                    )
            lpips_scores = batch_lpips(metrics["lpips"], seed_predictions, seed_targets)
            niqe_scores = batch_niqe(metrics["niqe"], seed_predictions)
            for row, lpips_score, niqe_score in zip(
                seed_rows, lpips_scores, niqe_scores, strict=True
            ):
                row["lpips"] = lpips_score
                row["niqe"] = niqe_score
            result_rows.extend(seed_rows)

            for dataset in sorted({row["dataset"] for row in rows}):
                dataset_row = next(row for row in rows if row["dataset"] == dataset)
                timing = timing_for_model(
                    model,
                    sources[track][dataset_row["low_path"]],
                    device,
                    warmup=args.warmup,
                    iterations=args.timing_iters,
                )
                model_timing[(track, seed, dataset)] = timing

    methods = ["cnn_unet"]
    summary_rows = summarize(
        result_rows,
        methods=methods,
        tracks=tracks,
        seeds=[*seeds, "all"],
    )
    seed_rows = seed_summary(summary_rows, methods=methods, tracks=tracks)

    comparison_rows: list[dict[str, object]] = []
    for track in tracks:
        for method in methods:
            for dataset in [*sorted({str(row["dataset"]) for row in rows}), "all"]:
                subset = [
                    row
                    for row in result_rows
                    if row["track"] == track
                    and row["method"] == method
                    and (dataset == "all" or row["dataset"] == dataset)
                ]
                for metric in METRIC_NAMES:
                    values = np.asarray([float(row[metric]) for row in subset])
                    scores = summarize_values(values)
                    comparison_rows.append(
                        {
                            "track": track,
                            "dataset": dataset,
                            "metric": metric,
                            "method": method,
                            **scores,
                        }
                    )

    classical_rows: list[dict[str, str]] = []
    frozen_best: dict[str, str] = {}
    if args.classical_per_image.exists():
        frozen_data = yaml.safe_load(args.frozen.read_text(encoding="utf-8"))
        frozen_best = {track: str(frozen_data["tracks"][track]["best_method"]) for track in tracks}
        classical_rows = read_csv_rows(args.classical_per_image)
        for track in tracks:
            for method in [frozen_best[track], "none"]:
                for dataset in [*sorted({str(row["dataset"]) for row in rows}), "all"]:
                    subset = [
                        row
                        for row in classical_rows
                        if row["track"] == track
                        and row["method"] == method
                        and (dataset == "all" or row["dataset"] == dataset)
                        and row["image_id"] in images
                    ]
                    if not subset:
                        continue
                    for metric in METRIC_NAMES:
                        values = np.asarray([float(row[metric]) for row in subset])
                        scores = summarize_values(values)
                        comparison_rows.append(
                            {
                                "track": track,
                                "dataset": dataset,
                                "metric": metric,
                                "method": f"classical_{method}",
                                **scores,
                            }
                        )

    test_rows = []
    for track in tracks:
        baselines = [f"classical_{frozen_best[track]}", "classical_none"] if frozen_best else []
        for baseline in baselines:
            baseline_method = baseline.removeprefix("classical_")
            baseline_rows = [
                row
                for row in classical_rows
                if row["track"] == track and row["method"] == baseline_method
            ]
            test_rows.extend(
                paired_tests(
                    result_rows,
                    baseline_rows,
                    track=track,
                    baseline=baseline,
                )
            )

    timing_rows = []
    for track in tracks:
        for seed in seeds:
            for dataset in sorted({str(row["dataset"]) for row in rows}):
                timing = model_timing[(track, seed, dataset)]
                source_values = [
                    source_ms[track][row["low_path"]] for row in rows if row["dataset"] == dataset
                ]
                timing_rows.append(
                    {
                        "track": track,
                        "method": "cnn_unet",
                        "dataset": dataset,
                        "device": device.type,
                        "seed": seed,
                        **timing,
                        "protocol": (f"batch=1 warmup={args.warmup} iters={args.timing_iters}"),
                        "source_median_ms": float(np.median(source_values)),
                    }
                )

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        per_image_path,
        [{key: format_cell(value) for key, value in row.items()} for row in result_rows],
        PER_IMAGE_COLUMNS,
    )
    write_csv_rows(
        out_dir / "summary.csv",
        [{key: format_cell(value) for key, value in row.items()} for row in summary_rows],
        SUMMARY_COLUMNS,
    )
    write_csv_rows(
        out_dir / "seed_summary.csv",
        [{key: format_cell(value) for key, value in row.items()} for row in seed_rows],
        SEED_SUMMARY_COLUMNS,
    )
    write_csv_rows(
        out_dir / "comparison.csv",
        [{key: format_cell(value) for key, value in row.items()} for row in comparison_rows],
        COMPARISON_COLUMNS,
    )
    if test_rows:
        write_csv_rows(
            out_dir / "comparison_tests.csv",
            [{key: format_cell(value) for key, value in row.items()} for row in test_rows],
            TEST_COLUMNS,
        )
    write_csv_rows(
        out_dir / "timing.csv",
        [{key: format_cell(value) for key, value in row.items()} for row in timing_rows],
        TIMING_COLUMNS,
    )

    split_manifest = json.loads(
        (ROOT / "data" / "splits" / "split_manifest.json").read_text(encoding="utf-8")
    )
    metadata = {
        "run_id": "stage05-cnn-brightening",
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "python": platform.python_version(),
        "device": str(device),
        "torch": torch.__version__,
        "test_used": is_test,
        "splits": list(args.splits),
        "limit": args.limit,
        "n_images": len(rows),
        "n_rows": len(result_rows),
        "tracks": tracks,
        "seeds": seeds,
        "checkpoint_root": str(args.checkpoint_root.relative_to(ROOT)),
        "checkpoints": checkpoint_meta,
        "training_config": str(
            (ROOT / "configs" / "training" / "brightening.yaml").relative_to(ROOT)
        ),
        "training_config_sha256": sha256(ROOT / "configs" / "training" / "brightening.yaml"),
        "model_config": str(
            (ROOT / "configs" / "models" / "unet_brightening.yaml").relative_to(ROOT)
        ),
        "model_config_sha256": sha256(ROOT / "configs" / "models" / "unet_brightening.yaml"),
        "d_fixed_config": str(args.d_fixed.relative_to(ROOT)),
        "d_fixed_config_sha256": sha256(args.d_fixed),
        "d_fixed": d_fixed_payload,
        "classical_per_image": str(args.classical_per_image.relative_to(ROOT))
        if args.classical_per_image.exists()
        else None,
        "classical_per_image_sha256": sha256(args.classical_per_image)
        if args.classical_per_image.exists()
        else None,
        "frozen_classical_config": str(args.frozen.relative_to(ROOT))
        if args.frozen.exists()
        else None,
        "frozen_classical_config_sha256": sha256(args.frozen) if args.frozen.exists() else None,
        "classical_best_method": frozen_best,
        "split_hashes": split_manifest.get("files_sha256", {}),
        "metrics": [metric.metadata() for metric in metrics.values()],
        "aggregation": "bootstrap 1000, ci 0.95, seed 0; non-finite excluded",
        "statistics": (
            "paired two-sided Wilcoxon signed-rank and paired t-test; CNN seed mean per image"
        ),
        "timing_protocol": f"batch=1 warmup={args.warmup} iters={args.timing_iters}",
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if is_test:
        lock_files = {
            "per_image_sha256": per_image_path,
            "summary_sha256": out_dir / "summary.csv",
            "seed_summary_sha256": out_dir / "seed_summary.csv",
            "comparison_sha256": out_dir / "comparison.csv",
            "comparison_tests_sha256": out_dir / "comparison_tests.csv",
            "timing_sha256": out_dir / "timing.csv",
        }
        lock = {
            "used_at_utc": metadata["created_at_utc"],
            "split_hashes": metadata["split_hashes"],
            **{key: sha256(path) for key, path in lock_files.items() if path.exists()},
        }
        (out_dir / "test_lock.json").write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")

    if not args.no_figures:
        figures = out_dir / "figures"
        figures.mkdir(exist_ok=True)
        save_validation_curves(
            figures / "validation_curves.png",
            ROOT / "results" / "brightening_cnn",
            tracks,
            seeds,
        )
        if classical_rows:
            frozen_data = yaml.safe_load(args.frozen.read_text(encoding="utf-8"))
            gamma = build_brightener(
                "gamma", frozen_data["tracks"]["d_fixed"]["methods"]["gamma"]["params"]
            )
        else:
            gamma = None
        if gamma is not None and rows:
            loaded = [(row, float(images[row["low_path"]].mean())) for row in rows]
            loaded.sort(key=lambda item: item[1])
            indices = sorted({len(loaded) // 6, len(loaded) // 2, 5 * len(loaded) // 6})
            examples = []
            for index in indices:
                row = loaded[index][0]
                image_id = row["low_path"]
                examples.append(
                    {
                        "dataset": row["dataset"],
                        "low": images[image_id],
                        "fixed": fixed_images[image_id],
                        "target": targets[image_id],
                        "raw_cnn": predictions[("raw", seeds[0], image_id)],
                        "fixed_cnn": predictions[("d_fixed", seeds[0], image_id)],
                    }
                )
            save_examples(figures / "brightening_examples.png", examples, gamma, seeds[0])
            save_detail(
                figures / "brightening_detail.png",
                examples[len(examples) // 2],
                gamma,
                seeds[0],
            )

            failures = []
            by_id = {row["low_path"]: row for row in rows}
            for dataset in sorted({str(row["dataset"]) for row in rows}):
                candidates = sorted(
                    [
                        row
                        for row in result_rows
                        if row["track"] == "d_fixed"
                        and row["seed"] == seeds[0]
                        and row["dataset"] == dataset
                    ],
                    key=lambda row: float(row["psnr"]),
                )
                seen_scenes: set[str] = set()
                picked = 0
                for candidate in candidates:
                    source_row = by_id[str(candidate["image_id"])]
                    scene = str(source_row["scene_id"])
                    if scene in seen_scenes:
                        continue
                    seen_scenes.add(scene)
                    image_id = str(candidate["image_id"])
                    failures.append(
                        {
                            "dataset": dataset,
                            "low": images[image_id],
                            "fixed": fixed_images[image_id],
                            "target": targets[image_id],
                            "fixed_cnn": predictions[("d_fixed", seeds[0], image_id)],
                        }
                    )
                    picked += 1
                    if picked == 3:
                        break
            if failures:
                save_failures(figures / "failure_cases.png", failures, gamma, seeds[0])
        print(f"figures={figures}")

    print(f"per_image={per_image_path}")
    print(f"summary={out_dir / 'summary.csv'}")
    print(f"seed_summary={out_dir / 'seed_summary.csv'}")
    print(f"comparison={out_dir / 'comparison.csv'}")
    print(f"timing={out_dir / 'timing.csv'}")
    print(f"metadata={out_dir / 'run_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
