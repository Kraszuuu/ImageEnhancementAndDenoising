"""Run metrics over image pairs and write the frozen CSV formats."""

from __future__ import annotations

import csv
import json
import math
import platform
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from lowlight.metrics.base import BaseMetric

PER_IMAGE_COLUMNS = [
    "run_id",
    "dataset",
    "split",
    "method",
    "seed",
    "noise_level",
    "image_id",
    "psnr",
    "ssim",
    "lpips",
    "niqe",
    "inference_ms",
    "width",
    "height",
]

SUMMARY_COLUMNS = [
    "method",
    "dataset",
    "split",
    "noise_level",
    "seed",
    "metric",
    "mean",
    "std",
    "median",
    "ci95_low",
    "ci95_high",
    "n",
]

METRIC_COLUMNS = ["psnr", "ssim", "lpips", "niqe"]
GROUP_COLUMNS = ["method", "dataset", "split", "noise_level"]
PACKAGES = ["numpy", "scipy", "scikit-image", "lpips", "pyiqa", "torch", "torchvision"]


@dataclass(frozen=True)
class ImagePair:
    image_id: str
    prediction: object
    target: object
    inference_ms: float | None = None


def _format_cell(value: object) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.6f}"
    return str(value)


def _package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _bootstrap_ci(
    values: np.ndarray, resamples: int, ci: float, rng: np.random.Generator
) -> tuple[float, float]:
    if values.size == 0:
        raise ValueError("bootstrap requires at least one value")
    if values.size == 1:
        return float(values[0]), float(values[0])
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    low, high = np.percentile(means, [alpha * 100.0, (1.0 - alpha) * 100.0])
    return float(low), float(high)


def _summarize_group(
    rows: Sequence[dict[str, object]],
    *,
    resamples: int,
    ci: float,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    output = []
    for metric in METRIC_COLUMNS:
        values = []
        for row in rows:
            raw = row.get(metric, "")
            if raw in ("", None):
                continue
            value = float(raw)
            if math.isfinite(value):
                values.append(value)
        if not values:
            output.append(
                {
                    **{key: rows[0][key] for key in GROUP_COLUMNS},
                    "seed": rows[0]["seed"],
                    "metric": metric,
                    "mean": "",
                    "std": "",
                    "median": "",
                    "ci95_low": "",
                    "ci95_high": "",
                    "n": 0,
                }
            )
            continue
        array = np.asarray(values, dtype=np.float64)
        low, high = _bootstrap_ci(array, resamples, ci, rng)
        std = float(array.std(ddof=1)) if array.size > 1 else 0.0
        output.append(
            {
                **{key: rows[0][key] for key in GROUP_COLUMNS},
                "seed": rows[0]["seed"],
                "metric": metric,
                "mean": float(array.mean()),
                "std": std,
                "median": float(np.median(array)),
                "ci95_low": low,
                "ci95_high": high,
                "n": int(array.size),
            }
        )
    return output


def summarize(
    rows: Sequence[dict[str, object]],
    *,
    resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Aggregate per-image rows per seed and pooled across seeds."""
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be in (0, 1)")
    if resamples < 1:
        raise ValueError("resamples must be positive")
    rng = np.random.default_rng(seed)

    def group(rows_subset: Sequence[dict[str, object]], seed_value: object):
        buckets: dict[tuple[str, ...], list[dict[str, object]]] = {}
        for row in rows_subset:
            key = tuple(str(row[column]) for column in GROUP_COLUMNS)
            buckets.setdefault(key, []).append(row)
        output = []
        for key in sorted(buckets):
            group_rows = [{**row, "seed": seed_value} for row in buckets[key]]
            output.extend(_summarize_group(group_rows, resamples=resamples, ci=ci, rng=rng))
        return output

    seeds = sorted({str(row["seed"]) for row in rows})
    summary = []
    for seed_key in seeds:
        subset = [row for row in rows if str(row["seed"]) == seed_key]
        summary.extend(group(subset, int(seed_key) if seed_key.isdigit() else seed_key))
    if len(seeds) > 1:
        summary.extend(group(rows, "all"))
    return summary


class MetricsRunner:
    """Evaluate a fixed metric set and write per-image, summary and metadata files."""

    def __init__(
        self,
        metrics: Iterable[BaseMetric],
        *,
        run_id: str,
        method: str,
        dataset: str,
        split: str,
        noise_level: str,
        seed: int = 0,
        split_hash: str | None = None,
        bootstrap_resamples: int = 1000,
        bootstrap_seed: int = 0,
        bootstrap_ci: float = 0.95,
    ) -> None:
        self.metrics: dict[str, BaseMetric] = {}
        for metric in metrics:
            if metric.name in self.metrics:
                raise ValueError(f"duplicate metric: {metric.name}")
            self.metrics[metric.name] = metric
        self.run_id = run_id
        self.method = method
        self.dataset = dataset
        self.split = split
        self.noise_level = noise_level
        self.seed = seed
        self.split_hash = split_hash
        self.bootstrap_resamples = bootstrap_resamples
        self.bootstrap_seed = bootstrap_seed
        self.bootstrap_ci = bootstrap_ci

    def evaluate(self, pairs: Iterable[ImagePair]) -> list[dict[str, object]]:
        rows = []
        for pair in pairs:
            shape = np.asarray(pair.target).shape
            row: dict[str, object] = {
                "run_id": self.run_id,
                "dataset": self.dataset,
                "split": self.split,
                "method": self.method,
                "seed": self.seed,
                "noise_level": self.noise_level,
                "image_id": pair.image_id,
                "width": int(shape[1]) if len(shape) >= 2 else 0,
                "height": int(shape[0]) if len(shape) >= 2 else 0,
                "inference_ms": "" if pair.inference_ms is None else float(pair.inference_ms),
            }
            for column in METRIC_COLUMNS:
                metric = self.metrics.get(column)
                row[column] = (
                    "" if metric is None else metric.compute(pair.prediction, pair.target)
                )
            rows.append(row)
        return rows

    def metadata(self, rows: Sequence[dict[str, object]]) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "method": self.method,
            "dataset": self.dataset,
            "split": self.split,
            "noise_level": self.noise_level,
            "seed": self.seed,
            "split_hash": self.split_hash,
            "n_images": len(rows),
            "per_image_file": "per_image.csv",
            "summary_file": "summary.csv",
            "metrics": [metric.metadata() for metric in self.metrics.values()],
            "package_versions": _package_versions(),
            "python": platform.python_version(),
            "deterministic": True,
            "aggregation": {
                "bootstrap_resamples": self.bootstrap_resamples,
                "bootstrap_seed": self.bootstrap_seed,
                "ci": self.bootstrap_ci,
                "note": "non-finite values are kept per-image but excluded from aggregates",
            },
        }

    def write(
        self, rows: Sequence[dict[str, object]], output_dir: Path
    ) -> tuple[Path, Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        per_image = output_dir / "per_image.csv"
        with per_image.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=PER_IMAGE_COLUMNS)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {column: _format_cell(row.get(column, "")) for column in PER_IMAGE_COLUMNS}
                )

        summary_rows = summarize(
            rows,
            resamples=self.bootstrap_resamples,
            ci=self.bootstrap_ci,
            seed=self.bootstrap_seed,
        )
        summary = output_dir / "summary.csv"
        with summary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(
                    {column: _format_cell(row.get(column, "")) for column in SUMMARY_COLUMNS}
                )

        metadata_path = output_dir / "run_metadata.json"
        metadata_path.write_text(
            json.dumps(self.metadata(rows), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return per_image, summary, metadata_path

    def run(self, pairs: Iterable[ImagePair], output_dir: Path) -> list[dict[str, object]]:
        rows = self.evaluate(pairs)
        self.write(rows, output_dir)
        return rows
