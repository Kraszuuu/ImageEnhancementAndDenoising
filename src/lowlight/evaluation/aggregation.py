"""Normalize per-image result CSVs and aggregate them for Stage 07."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lowlight.evaluation.statistics import bootstrap_ci
from lowlight.utils.io import read_csv_rows

METRIC_COLUMNS = ["psnr", "ssim", "lpips", "niqe"]
FAILURE_COLUMNS = [
    "brightness_error",
    "color_cast",
    "clip_low_fraction",
    "clip_high_fraction",
]
UNIFIED_COLUMNS = [
    "source",
    "task",
    "variant",
    "method",
    "track",
    "dataset",
    "split",
    "noise_level",
    "seed",
    "image_id",
    "psnr",
    "ssim",
    "lpips",
    "niqe",
    "inference_ms",
    *FAILURE_COLUMNS,
]


@dataclass(frozen=True)
class SourceSpec:
    """Description of one existing per-image CSV."""

    name: str
    path: Path
    task: str
    variant: str
    method_column: str = "method"
    default_noise_level: str = ""
    default_track: str = ""


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0.0 else "-inf"
    return str(value)


def load_source(spec: SourceSpec) -> list[dict[str, str]]:
    """Load one source and map it to the unified Stage 07 schema."""
    rows = read_csv_rows(spec.path)
    output: list[dict[str, str]] = []
    for row in rows:
        method = row.get(spec.method_column, "").strip()
        if not method:
            raise ValueError(f"{spec.path}: missing method in column {spec.method_column!r}")
        normalized = {
            "source": spec.name,
            "task": spec.task,
            "variant": spec.variant,
            "method": method,
            "track": row.get("track", spec.default_track),
            "dataset": row.get("dataset", ""),
            "split": row.get("split", ""),
            "noise_level": row.get("noise_level", spec.default_noise_level),
            "seed": row.get("seed", "0") or "0",
            "image_id": row.get("image_id", ""),
            "psnr": row.get("psnr", ""),
            "ssim": row.get("ssim", ""),
            "lpips": row.get("lpips", ""),
            "niqe": row.get("niqe", ""),
            "inference_ms": row.get("inference_ms", row.get("total_ms", "")),
            "brightness_error": row.get("brightness_error", ""),
            "color_cast": row.get("color_cast", ""),
            "clip_low_fraction": row.get("clip_low_fraction", ""),
            "clip_high_fraction": row.get("clip_high_fraction", ""),
        }
        output.append(normalized)
    return output


def load_sources(specs: Iterable[SourceSpec]) -> list[dict[str, str]]:
    rows = []
    for spec in specs:
        rows.extend(load_source(spec))
    if not rows:
        raise ValueError("no rows loaded from sources")
    return rows


def filter_rows(
    rows: Sequence[dict[str, str]],
    *,
    task: str | None = None,
    variant: str | None = None,
    method: str | None = None,
    track: str | None = None,
    dataset: str | None = None,
    noise_level: str | None = None,
) -> list[dict[str, str]]:
    filters = {
        "task": task,
        "variant": variant,
        "method": method,
        "track": track,
        "dataset": dataset,
        "noise_level": noise_level,
    }
    return [
        row
        for row in rows
        if all(value is None or row.get(key, "") == value for key, value in filters.items())
    ]


def _float(value: object) -> float:
    try:
        parsed = float(str(value))
    except TypeError, ValueError:
        return math.nan
    return parsed if math.isfinite(parsed) else math.nan


def per_image_mean(rows: Sequence[dict[str, str]], metric: str) -> dict[tuple[str, str], float]:
    """Average a metric over seeds for each ``(dataset, image_id)`` pair."""
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = _float(row.get(metric, ""))
        if math.isfinite(value):
            buckets[(row.get("dataset", ""), row.get("image_id", ""))].append(value)
    return {key: float(np.mean(values)) for key, values in buckets.items()}


def aggregate_rows(
    rows: Sequence[dict[str, str]],
    group_by: Sequence[str],
    *,
    metrics: Sequence[str] = METRIC_COLUMNS,
    resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Aggregate quality metrics over images and seeds.

    The primary ``mean``/``std``/CI use one value per image (seed mean first),
    so three seeds do not pretend to be three independent test images.
    ``seed_mean``/``seed_std`` report variability between training seeds.
    """
    buckets: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(str(row.get(key, "")) for key in group_by)].append(row)

    output: list[dict[str, object]] = []
    for group_index, (key, group_rows) in enumerate(sorted(buckets.items())):
        base = dict(zip(group_by, key, strict=True))
        seeds = sorted({row.get("seed", "0") for row in group_rows})
        per_seed: dict[str, dict[tuple[str, str], float]] = {}
        for seed_value in seeds:
            seed_rows = [row for row in group_rows if row.get("seed", "0") == seed_value]
            for metric in metrics:
                per_seed.setdefault(f"{seed_value}:{metric}", {}).update(
                    per_image_mean(seed_rows, metric)
                )

        for metric in metrics:
            per_image_values: dict[tuple[str, str], list[float]] = defaultdict(list)
            seed_means: list[float] = []
            for seed_value in seeds:
                values = per_seed.get(f"{seed_value}:{metric}", {})
                values = [value for value in values.values() if math.isfinite(value)]
                if values:
                    seed_means.append(float(np.mean(values)))
                for image_key, value in per_seed.get(f"{seed_value}:{metric}", {}).items():
                    per_image_values[image_key].append(value)

            values = [
                float(np.mean(seed_values))
                for seed_values in per_image_values.values()
                if seed_values
            ]
            if not values:
                output.append(
                    {
                        **base,
                        "metric": metric,
                        "mean": "",
                        "std": "",
                        "median": "",
                        "ci95_low": "",
                        "ci95_high": "",
                        "seed_mean": "",
                        "seed_std": "",
                        "n": 0,
                        "n_seeds": len(seeds),
                    }
                )
                continue

            array = np.asarray(values, dtype=np.float64)
            low, high = bootstrap_ci(array, resamples=resamples, ci=ci, seed=seed + group_index)
            seed_array = np.asarray(seed_means, dtype=np.float64)
            seed_std = float(seed_array.std(ddof=1)) if seed_array.size > 1 else 0.0
            output.append(
                {
                    **base,
                    "metric": metric,
                    "mean": float(array.mean()),
                    "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
                    "median": float(np.median(array)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "seed_mean": float(seed_array.mean()) if seed_array.size else "",
                    "seed_std": seed_std if seed_array.size else "",
                    "n": int(array.size),
                    "n_seeds": len(seeds),
                }
            )
    return output
