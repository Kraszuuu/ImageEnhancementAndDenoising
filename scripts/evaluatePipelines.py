"""Evaluate frozen Stage 06 combined pipelines on the locked test splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lowlight.metrics import (  # noqa: E402
    LpipsMetric,
    NiqeMetric,
    PsnrMetric,
    SsimMetric,
    resolve_device,
)
from lowlight.pipelines import build_pipeline  # noqa: E402
from lowlight.utils.io import load_rgb, read_csv_rows, write_csv_rows  # noqa: E402
from lowlight.utils.noise import load_noise_config  # noqa: E402

TEST_SPLITS = ("lol_test.csv", "lolv2_test.csv")
METRICS = ("psnr", "ssim", "lpips", "niqe")
ALL_LEVELS = ("real", "low", "medium", "high")

PIPELINE_SPECS: dict[str, dict[str, object]] = {
    "classical_classical": {
        "config": "configs/pipelines/classical_classical.yaml",
        "levels": ALL_LEVELS,
        "seeded": False,
    },
    "brighten_then_denoise": {
        "config": "configs/pipelines/brighten_then_denoise.yaml",
        "levels": ALL_LEVELS,
        "seeded": False,
    },
    "classical_cnn": {
        "config": "configs/pipelines/classical_cnn.yaml",
        "levels": ALL_LEVELS,
        "seeded": True,
    },
    "cnn_classical": {
        "config": "configs/pipelines/cnn_classical.yaml",
        "levels": ALL_LEVELS,
        "seeded": True,
    },
    "cnn_cnn": {
        "config": "configs/pipelines/cnn_cnn.yaml",
        "levels": ALL_LEVELS,
        "seeded": True,
    },
    "end_to_end_real": {
        "config": "configs/pipelines/end_to_end.yaml",
        "variant": "real",
        "levels": ("real",),
        "seeded": True,
    },
    "end_to_end_controlled": {
        "config": "configs/pipelines/end_to_end.yaml",
        "variant": "controlled",
        "levels": ("low", "medium", "high"),
        "seeded": True,
    },
}

PER_IMAGE_COLUMNS = [
    "run_id",
    "pipeline",
    "dataset",
    "split",
    "seed",
    "noise_level",
    "image_id",
    "psnr",
    "ssim",
    "lpips",
    "niqe",
    "total_ms",
    "denoise_ms",
    "brighten_ms",
    "end_to_end_ms",
    "width",
    "height",
]
SUMMARY_COLUMNS = [
    "pipeline",
    "seed",
    "dataset",
    "noise_level",
    "metric",
    "mean",
    "std",
    "median",
    "ci95_low",
    "ci95_high",
    "n",
]
SEED_SUMMARY_COLUMNS = [
    "pipeline",
    "dataset",
    "noise_level",
    "metric",
    "seed_mean",
    "seed_std",
    "n_seeds",
]
TIMING_COLUMNS = [
    "pipeline",
    "component",
    "dataset",
    "noise_level",
    "device",
    "seed",
    "mean_ms",
    "std_ms",
    "median_ms",
    "n",
    "protocol",
]
TEST_COLUMNS = [
    "noise_level",
    "metric",
    "pipeline_a",
    "pipeline_b",
    "n_pairs",
    "mean_a",
    "mean_b",
    "mean_delta",
    "median_delta",
    "effect_size_dz",
    "wilcoxon_stat",
    "wilcoxon_p",
    "ttest_stat",
    "ttest_p",
    "p_holm",
    "significant_holm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipelines",
        nargs="+",
        choices=tuple(PIPELINE_SPECS),
        default=list(PIPELINE_SPECS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--splits", nargs="*", default=list(TEST_SPLITS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--noise-levels", nargs="+", choices=ALL_LEVELS, default=list(ALL_LEVELS))
    parser.add_argument(
        "--noise-config", type=Path, default=ROOT / "configs" / "data" / "noise.yaml"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "results" / "pipelines" / "evaluation"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def format_cell(value: object) -> object:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
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
    pipelines: list[str],
    seeds: list[int],
) -> list[dict[str, object]]:
    datasets = sorted({str(row["dataset"]) for row in rows})
    levels = sorted({str(row["noise_level"]) for row in rows})
    output = []
    for pipeline in pipelines:
        pipeline_seeds = sorted({int(row["seed"]) for row in rows if row["pipeline"] == pipeline})
        if not pipeline_seeds:
            continue
        seed_values: list[int | str] = (
            [*pipeline_seeds, "all"] if len(pipeline_seeds) > 1 else pipeline_seeds
        )
        for seed in seed_values:
            selected = [
                row
                for row in rows
                if row["pipeline"] == pipeline and (seed == "all" or row["seed"] == seed)
            ]
            if not selected:
                continue
            dataset_values = [*datasets, "all"] if len(datasets) > 1 else datasets
            for dataset in dataset_values:
                dataset_rows = (
                    selected
                    if dataset == "all"
                    else [row for row in selected if row["dataset"] == dataset]
                )
                level_values = [*levels, "all"] if len(levels) > 1 else levels
                for level in level_values:
                    subset = (
                        dataset_rows
                        if level == "all"
                        else [row for row in dataset_rows if row["noise_level"] == level]
                    )
                    if not subset:
                        continue
                    for metric in METRICS:
                        values = np.asarray([float(row[metric]) for row in subset])
                        output.append(
                            {
                                "pipeline": pipeline,
                                "seed": seed,
                                "dataset": dataset,
                                "noise_level": level,
                                "metric": metric,
                                **summarize_values(values),
                            }
                        )
    return output


def seed_summary(
    summary_rows: list[dict[str, object]],
    *,
    pipelines: list[str],
) -> list[dict[str, object]]:
    output = []
    for pipeline in pipelines:
        for dataset in ["all"]:
            for level in sorted(
                {str(row["noise_level"]) for row in summary_rows if row["pipeline"] == pipeline}
            ):
                for metric in METRICS:
                    per_seed = [
                        float(row["mean"])
                        for row in summary_rows
                        if row["pipeline"] == pipeline
                        and row["dataset"] == dataset
                        and row["noise_level"] == level
                        and row["metric"] == metric
                        and row["seed"] != "all"
                    ]
                    if not per_seed:
                        continue
                    values = np.asarray(per_seed)
                    output.append(
                        {
                            "pipeline": pipeline,
                            "dataset": dataset,
                            "noise_level": level,
                            "metric": metric,
                            "seed_mean": float(values.mean()),
                            "seed_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                            "n_seeds": int(values.size),
                        }
                    )
    return output


def per_image_seed_means(
    rows: list[dict[str, object]],
) -> dict[tuple[str, str, str], dict[str, float]]:
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        key = (str(row["pipeline"]), str(row["noise_level"]), str(row["image_id"]))
        for metric in METRICS:
            grouped[key][metric].append(float(row[metric]))
    return {
        key: {metric: float(np.mean(values)) for metric, values in metrics.items()}
        for key, metrics in grouped.items()
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    previous = 0.0
    for rank, index in enumerate(order, start=1):
        value = min(1.0, p_values[index] * (count - rank + 1))
        previous = max(previous, value)
        adjusted[index] = previous
    return adjusted


def paired_tests(
    rows: list[dict[str, object]],
    *,
    pipelines: list[str],
) -> list[dict[str, object]]:
    try:
        from scipy.stats import ttest_rel, wilcoxon
    except ImportError:
        return []

    by_method = per_image_seed_means(rows)
    levels = sorted({str(row["noise_level"]) for row in rows})
    output = []
    for level in levels:
        for metric in METRICS:
            family = []
            for index_a, pipeline_a in enumerate(pipelines):
                for pipeline_b in pipelines[index_a + 1 :]:
                    pairs = []
                    for (pipeline, row_level, image_id), values in by_method.items():
                        if pipeline != pipeline_a or row_level != level:
                            continue
                        other = by_method.get((pipeline_b, level, image_id))
                        if other is None:
                            continue
                        value_a = values[metric]
                        value_b = other[metric]
                        if math.isfinite(value_a) and math.isfinite(value_b):
                            pairs.append((value_a, value_b))
                    if len(pairs) < 2:
                        continue
                    values_a = np.asarray([pair[0] for pair in pairs])
                    values_b = np.asarray([pair[1] for pair in pairs])
                    differences = values_a - values_b
                    std = float(differences.std(ddof=1))
                    effect = float(differences.mean() / std) if std > 0 else 0.0
                    try:
                        wilcoxon_result = wilcoxon(differences)
                        wilcoxon_stat = float(wilcoxon_result.statistic)
                        wilcoxon_p = float(wilcoxon_result.pvalue)
                    except ValueError:
                        wilcoxon_stat = math.nan
                        wilcoxon_p = math.nan
                    ttest_result = ttest_rel(values_a, values_b)
                    ttest_p = float(ttest_result.pvalue)
                    if not math.isfinite(ttest_p):
                        ttest_p = 1.0
                    if not math.isfinite(wilcoxon_p):
                        wilcoxon_p = 1.0
                    family.append(
                        {
                            "noise_level": level,
                            "metric": metric,
                            "pipeline_a": pipeline_a,
                            "pipeline_b": pipeline_b,
                            "n_pairs": len(pairs),
                            "mean_a": float(values_a.mean()),
                            "mean_b": float(values_b.mean()),
                            "mean_delta": float(differences.mean()),
                            "median_delta": float(np.median(differences)),
                            "effect_size_dz": effect,
                            "wilcoxon_stat": wilcoxon_stat,
                            "wilcoxon_p": wilcoxon_p,
                            "ttest_stat": float(ttest_result.statistic),
                            "ttest_p": ttest_p,
                        }
                    )
            if family:
                adjusted = holm_adjust([float(row["ttest_p"]) for row in family])
                for row, p_value in zip(family, adjusted, strict=True):
                    row["p_holm"] = p_value
                    row["significant_holm"] = p_value < 0.05
                output.extend(family)
    return output


def metric_values(
    metrics: dict[str, object], prediction: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    return {
        name: float(metric.compute(prediction, target))  # type: ignore[attr-defined]
        for name, metric in metrics.items()
    }


def batch_lpips(
    metric: LpipsMetric,
    predictions: list[np.ndarray],
    targets: list[np.ndarray],
    batch: int = 8,
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
        with torch.inference_mode():
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


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def collect_checkpoints(pipeline_metadata: list[dict[str, object]]) -> dict[str, str]:
    checkpoints: dict[str, str] = {}
    for metadata in pipeline_metadata:
        for step in metadata.get("steps", []):  # type: ignore[union-attr]
            component = step.get("component", {})  # type: ignore[union-attr]
            path = component.get("checkpoint")
            if path:
                checkpoint_path = ROOT / str(path)
                if checkpoint_path.exists():
                    checkpoints[str(path)] = sha256(checkpoint_path)
    return checkpoints


def timing_rows(
    rows: list[dict[str, object]],
    *,
    pipelines: list[str],
    seeds: list[int],
    device: torch.device,
) -> list[dict[str, object]]:
    output = []
    components = ("total_ms", "denoise_ms", "brighten_ms", "end_to_end_ms")
    for pipeline in pipelines:
        for seed in seeds:
            selected = [row for row in rows if row["pipeline"] == pipeline and row["seed"] == seed]
            if not selected:
                continue
            datasets = sorted({str(row["dataset"]) for row in selected})
            levels = sorted({str(row["noise_level"]) for row in selected})
            dataset_values = [*datasets, "all"] if len(datasets) > 1 else datasets
            level_values = [*levels, "all"] if len(levels) > 1 else levels
            for dataset in dataset_values:
                dataset_rows = (
                    selected
                    if dataset == "all"
                    else [row for row in selected if row["dataset"] == dataset]
                )
                for level in level_values:
                    subset = (
                        dataset_rows
                        if level == "all"
                        else [row for row in dataset_rows if row["noise_level"] == level]
                    )
                    for component in components:
                        values = [
                            float(row[component])
                            for row in subset
                            if row.get(component, "") not in ("", None)
                        ]
                        if not values:
                            continue
                        array = np.asarray(values)
                        output.append(
                            {
                                "pipeline": pipeline,
                                "component": component.removesuffix("_ms"),
                                "dataset": dataset,
                                "noise_level": level,
                                "device": device.type,
                                "seed": seed,
                                "mean_ms": float(array.mean()),
                                "std_ms": float(array.std(ddof=1)) if array.size > 1 else 0.0,
                                "median_ms": float(np.median(array)),
                                "n": int(array.size),
                                "protocol": "per-image pipeline timing with CUDA sync",
                            }
                        )
    return output


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    per_image_path = out_dir / "per_image.csv"
    if args.limit > 0 and tuple(args.splits) == TEST_SPLITS:
        raise SystemExit("--limit is forbidden on locked test splits")
    is_test = tuple(args.splits) == TEST_SPLITS and args.limit == 0
    lock_path = out_dir / "test_lock.json"
    if (per_image_path.exists() or (is_test and lock_path.exists())) and not args.force:
        raise SystemExit(
            f"refusing to overwrite {out_dir}; test is locked "
            "(use --force only with a documented reason)"
        )
    selected_pipelines = [str(name) for name in args.pipelines]
    seeds = [int(seed) for seed in args.seeds]
    requested_levels = [str(level) for level in args.noise_levels]
    if is_test:
        for name in selected_pipelines:
            config_path = ROOT / str(PIPELINE_SPECS[name]["config"])
            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if not payload.get("frozen"):
                raise SystemExit(f"pipeline config is not frozen: {config_path}")

    rows = load_rows(args.splits, args.limit)
    device = resolve_device(None if args.device == "auto" else args.device)
    metrics = {"psnr": PsnrMetric(), "ssim": SsimMetric()}
    lpips_metric = LpipsMetric(device=device)
    niqe_metric = NiqeMetric(device=device)
    images = {row["low_path"]: load_rgb(ROOT / row["low_path"]) for row in rows}
    targets = {row["low_path"]: load_rgb(ROOT / row["gt_path"]) for row in rows}
    noise_config = load_noise_config(args.noise_config)
    controlled_levels = [level for level in ALL_LEVELS if level != "real"]
    noisy_images: dict[str, dict[str, np.ndarray]] = {level: {} for level in controlled_levels}
    for level in controlled_levels:
        for row in rows:
            image_id = row["low_path"]
            noisy_images[level][image_id] = noise_config.add(
                images[image_id], level, image_id=image_id, seed=0
            )
    sources = {"real": images, **noisy_images}

    needs_fixed = any(
        name in {"classical_classical", "classical_cnn"} for name in selected_pipelines
    )
    fixed_outputs: dict[tuple[str, str], np.ndarray] = {}
    fixed_ms: dict[tuple[str, str], float] = {}
    if needs_fixed:
        fixed_pipeline = build_pipeline(
            "configs/pipelines/classical_classical.yaml", root=ROOT, seed=0, device=device
        )
        fixed_component = fixed_pipeline.steps[0][1]
        fixed_levels = [
            level for level in requested_levels if level in {"real", "low", "medium", "high"}
        ]
        print(f"precomputing D_fixed for {len(rows)} images x {len(fixed_levels)} levels")
        for level in fixed_levels:
            for index, row in enumerate(rows, start=1):
                image_id = row["low_path"]
                started = time.perf_counter()
                output = fixed_component(sources[level][image_id])
                fixed_ms[(level, image_id)] = (time.perf_counter() - started) * 1000.0
                fixed_outputs[(level, image_id)] = output.astype(np.float16)
                if index % 20 == 0 or index == len(rows):
                    print(f"  D_fixed {level} {index}/{len(rows)}")

    result_rows: list[dict[str, object]] = []
    pipeline_metadata: list[dict[str, object]] = []
    for name in selected_pipelines:
        spec = PIPELINE_SPECS[name]
        levels = [level for level in spec["levels"] if level in requested_levels]  # type: ignore[union-attr]
        if not levels:
            continue
        pipeline_seeds = seeds if bool(spec["seeded"]) else [0]
        for seed in pipeline_seeds:
            pipeline = build_pipeline(
                str(spec["config"]),
                root=ROOT,
                seed=seed,
                device=device,
                variant=str(spec["variant"]) if "variant" in spec else None,
            )
            pipeline_info = pipeline.metadata()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                pipeline_info["vram"] = {
                    "allocated_mib": round(torch.cuda.memory_allocated(device) / 2**20, 1),
                    "reserved_mib": round(torch.cuda.memory_reserved(device) / 2**20, 1),
                    "peak_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
                }
            pipeline_metadata.append(pipeline_info)
            print(f"pipeline={name} seed={seed} levels={levels}")
            for level in levels:
                level_rows: list[dict[str, object]] = []
                predictions: list[np.ndarray] = []
                level_targets: list[np.ndarray] = []
                for row in rows:
                    image_id = row["low_path"]
                    target = targets[image_id]
                    source = sources[level][image_id]
                    if name in {"classical_classical", "classical_cnn"}:
                        fixed = fixed_outputs[(level, image_id)].astype(np.float64)
                        brighten = pipeline.steps[1][1]
                        started = time.perf_counter()
                        prediction = brighten(fixed)
                        brighten_ms = (time.perf_counter() - started) * 1000.0
                        denoise_ms = fixed_ms[(level, image_id)]
                        total_ms = denoise_ms + brighten_ms
                        end_to_end_ms = None
                    else:
                        output = pipeline.run_timed(source)
                        prediction = output.image
                        denoise_ms = output.component_ms.get("denoise")
                        brighten_ms = output.component_ms.get("brighten")
                        end_to_end_ms = output.component_ms.get("end_to_end")
                        total_ms = output.total_ms
                    scores = metric_values(metrics, prediction, target)
                    level_rows.append(
                        {
                            "run_id": "stage06-pipeline-evaluation",
                            "pipeline": name,
                            "dataset": row["dataset"],
                            "split": row["split"],
                            "seed": seed,
                            "noise_level": level,
                            "image_id": image_id,
                            "psnr": scores["psnr"],
                            "ssim": scores["ssim"],
                            "lpips": 0.0,
                            "niqe": 0.0,
                            "total_ms": total_ms,
                            "denoise_ms": denoise_ms if denoise_ms is not None else "",
                            "brighten_ms": brighten_ms if brighten_ms is not None else "",
                            "end_to_end_ms": end_to_end_ms if end_to_end_ms is not None else "",
                            "width": target.shape[1],
                            "height": target.shape[0],
                        }
                    )
                    predictions.append(prediction)
                    level_targets.append(target)
                lpips_scores = batch_lpips(
                    lpips_metric, predictions, level_targets, batch=args.batch_size
                )
                niqe_scores = batch_niqe(niqe_metric, predictions, batch=args.batch_size)
                for row, lpips_score, niqe_score in zip(
                    level_rows, lpips_scores, niqe_scores, strict=True
                ):
                    row["lpips"] = lpips_score
                    row["niqe"] = niqe_score
                result_rows.extend(level_rows)
                print(
                    f"  {name} s{seed} {level}: "
                    f"PSNR={np.mean([float(r['psnr']) for r in level_rows]):.3f}"
                )

    summary_rows = summarize(result_rows, pipelines=selected_pipelines, seeds=seeds)
    seed_rows = seed_summary(summary_rows, pipelines=selected_pipelines)
    test_rows = paired_tests(result_rows, pipelines=selected_pipelines)
    timing = timing_rows(
        result_rows,
        pipelines=selected_pipelines,
        seeds=seeds,
        device=device,
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
        out_dir / "timing.csv",
        [{key: format_cell(value) for key, value in row.items()} for row in timing],
        TIMING_COLUMNS,
    )
    write_csv_rows(
        out_dir / "comparison_tests.csv",
        [{key: format_cell(value) for key, value in row.items()} for row in test_rows],
        TEST_COLUMNS,
    )

    split_manifest = json.loads(
        (ROOT / "data" / "splits" / "split_manifest.json").read_text(encoding="utf-8")
    )
    checkpoints = collect_checkpoints(pipeline_metadata)
    metadata = {
        "run_id": "stage06-pipeline-evaluation",
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
        "pipelines": selected_pipelines,
        "seeds": seeds,
        "noise_levels": requested_levels,
        "pipeline_configs": {
            name: {
                "path": str(PIPELINE_SPECS[name]["config"]),
                "sha256": sha256(ROOT / str(PIPELINE_SPECS[name]["config"])),
            }
            for name in selected_pipelines
        },
        "pipeline_metadata": pipeline_metadata,
        "checkpoints": checkpoints,
        "peak_vram_mib": (
            round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
            if device.type == "cuda"
            else None
        ),
        "noise_config_sha256": sha256(args.noise_config),
        "split_hashes": split_manifest.get("files_sha256", {}),
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    outputs = {
        "per_image.csv": sha256(per_image_path),
        "summary.csv": sha256(out_dir / "summary.csv"),
        "seed_summary.csv": sha256(out_dir / "seed_summary.csv"),
        "timing.csv": sha256(out_dir / "timing.csv"),
        "comparison_tests.csv": sha256(out_dir / "comparison_tests.csv"),
    }
    if is_test:
        lock = {
            "used_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "split_hashes": split_manifest.get("files_sha256", {}),
            "checkpoints": checkpoints,
            "outputs": outputs,
        }
        lock_path.write_text(
            json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(f"per_image={per_image_path}")
    print(f"rows={len(result_rows)} summary={len(summary_rows)} tests={len(test_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
