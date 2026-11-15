"""Grid tuning for classical denoisers on validation splits."""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from lowlight.classical.denoisers import BaseDenoiser, build_denoiser
from lowlight.metrics.base import BaseMetric
from lowlight.utils.noise import NoiseConfig

TUNING_LEVELS = ("clean", "low", "medium", "high")
NOISY_LEVELS = ("low", "medium", "high")
TUNING_COLUMNS = [
    "method",
    "level",
    "params",
    "n",
    "mean_psnr",
    "std_psnr",
    "mean_ssim",
    "std_ssim",
    "mean_ms",
    "median_ms",
]


@dataclass(frozen=True)
class TuningSample:
    dataset: str
    split: str
    image_id: str
    clean: np.ndarray


def iter_grid(grid: dict[str, Sequence[object]]) -> list[dict[str, object]]:
    """Deterministic cartesian product; keys are sorted."""
    keys = sorted(grid)
    values = [tuple(grid[key]) for key in keys]
    return [dict(zip(keys, combination, strict=True)) for combination in itertools.product(*values)]


def params_key(params: dict[str, object]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def prepare_noisy(
    samples: Sequence[TuningSample],
    noise_config: NoiseConfig,
    levels: Sequence[str],
    seed: int,
) -> dict[str, list[np.ndarray]]:
    return {
        level: [
            noise_config.add(
                sample.clean, level, image_id=sample.image_id, seed=seed
            )
            for sample in samples
        ]
        for level in levels
    }


def evaluate_config(
    method: str,
    params: dict[str, object],
    noisy_by_level: dict[str, list[np.ndarray]],
    metrics: dict[str, BaseMetric],
    targets: Sequence[np.ndarray],
) -> list[dict[str, object]]:
    """Evaluate one parameter set on cached noisy inputs against clean targets."""
    denoiser: BaseDenoiser = build_denoiser(method, params)
    rows: list[dict[str, object]] = []
    values_by_level: dict[str, dict[str, list[float]]] = {}
    durations_by_level: dict[str, list[float]] = {}
    for level, images in noisy_by_level.items():
        if len(images) != len(targets):
            raise ValueError("noisy and target counts differ")
        values: dict[str, list[float]] = {name: [] for name in metrics}
        durations = []
        for noisy, target in zip(images, targets, strict=True):
            start = time.perf_counter()
            denoised = denoiser.denoise(noisy)
            durations.append((time.perf_counter() - start) * 1000.0)
            for name, metric in metrics.items():
                values[name].append(metric.compute(denoised, target))
        rows.append(_aggregate_row(method, level, params, values, durations))
        values_by_level[level] = values
        durations_by_level[level] = durations

    def pooled(levels: Sequence[str]) -> tuple[dict[str, list[float]], list[float]]:
        merged: dict[str, list[float]] = {name: [] for name in metrics}
        times: list[float] = []
        for level in levels:
            if level not in values_by_level:
                continue
            for name in metrics:
                merged[name].extend(values_by_level[level][name])
            times.extend(durations_by_level[level])
        return merged, times

    all_values, all_times = pooled(list(noisy_by_level))
    rows.append(_aggregate_row(method, "all", params, all_values, all_times))
    noisy_values, noisy_times = pooled(list(NOISY_LEVELS))
    if noisy_times:
        rows.append(
            _aggregate_row(method, "noisy", params, noisy_values, noisy_times)
        )
    return rows


def _aggregate_row(
    method: str,
    level: str,
    params: dict[str, object],
    values: dict[str, list[float]],
    durations: Sequence[float],
) -> dict[str, object]:
    return {
        "method": method,
        "level": level,
        "params": params_key(params),
        "n": len(durations),
        "mean_psnr": float(np.mean(values["psnr"])),
        "std_psnr": float(np.std(values["psnr"], ddof=1)),
        "mean_ssim": float(np.mean(values["ssim"])),
        "std_ssim": float(np.std(values["ssim"], ddof=1)),
        "mean_ms": float(np.mean(durations)),
        "median_ms": float(np.median(durations)),
    }


def best_row(rows: Iterable[dict[str, object]], level: str) -> dict[str, object]:
    candidates = [row for row in rows if row["level"] == level]
    if not candidates:
        raise ValueError(f"no tuning rows for level {level!r}")
    return min(
        candidates,
        key=lambda row: (
            -float(row["mean_psnr"]),
            -float(row["mean_ssim"]),
            float(row["median_ms"]),
        ),
    )


def select_per_level(
    rows: Sequence[dict[str, object]],
    methods: Sequence[str],
    levels: Sequence[str] = TUNING_LEVELS,
) -> dict[str, dict[str, dict[str, object]]]:
    selected: dict[str, dict[str, dict[str, object]]] = {}
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        selected[method] = {
            level: best_row(method_rows, level) for level in levels
        }
    return selected


def select_fixed(
    rows: Sequence[dict[str, object]],
    candidates: Sequence[str] = ("bm3d", "nlm"),
) -> dict[str, object]:
    eligible = [
        row for row in rows if row["method"] in candidates and row["level"] == "noisy"
    ]
    if not eligible:
        raise ValueError("no eligible D_fixed candidates")
    return min(
        eligible,
        key=lambda row: (
            -float(row["mean_psnr"]),
            -float(row["mean_ssim"]),
            float(row["median_ms"]),
        ),
    )
