"""Grid tuning helpers for classical brighteners."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from lowlight.classical.brighteners import BaseBrightener, build_brightener
from lowlight.metrics.base import BaseMetric

BRIGHTENING_COLUMNS = [
    "track",
    "method",
    "params",
    "n",
    "mean_psnr",
    "std_psnr",
    "mean_ssim",
    "std_ssim",
    "mean_lpips",
    "std_lpips",
    "mean_ms",
    "median_ms",
]


@dataclass(frozen=True)
class BrighteningSample:
    dataset: str
    split: str
    image_id: str
    target: np.ndarray


def params_key(params: dict[str, object]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def evaluate_brightener(
    track: str,
    method: str,
    params: dict[str, object],
    sources: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    metrics: dict[str, BaseMetric],
) -> dict[str, object]:
    """Evaluate one parameter set on one track against clean targets."""
    if len(sources) != len(targets):
        raise ValueError("source and target counts differ")
    brightener: BaseBrightener = build_brightener(method, params)
    values: dict[str, list[float]] = {name: [] for name in metrics}
    durations: list[float] = []
    for source, target in zip(sources, targets, strict=True):
        start = time.perf_counter()
        prediction = brightener.enhance(source)
        durations.append((time.perf_counter() - start) * 1000.0)
        for name, metric in metrics.items():
            values[name].append(metric.compute(prediction, target))

    return {
        "track": track,
        "method": method,
        "params": params_key(params),
        "n": len(durations),
        "mean_psnr": float(np.mean(values["psnr"])),
        "std_psnr": float(np.std(values["psnr"], ddof=1)),
        "mean_ssim": float(np.mean(values["ssim"])),
        "std_ssim": float(np.std(values["ssim"], ddof=1)),
        "mean_lpips": float(np.mean(values["lpips"])),
        "std_lpips": float(np.std(values["lpips"], ddof=1)),
        "mean_ms": float(np.mean(durations)),
        "median_ms": float(np.median(durations)),
    }


def select_per_track(
    rows: Iterable[dict[str, object]],
    tracks: Sequence[str],
    methods: Sequence[str],
) -> dict[str, dict[str, dict[str, object]]]:
    """Select best parameters by PSNR, SSIM, LPIPS, then time."""
    rows = list(rows)
    selected: dict[str, dict[str, dict[str, object]]] = {}
    for track in tracks:
        selected[track] = {}
        for method in methods:
            candidates = [
                row
                for row in rows
                if row["track"] == track and row["method"] == method
            ]
            if not candidates:
                raise ValueError(f"no tuning rows for {track}/{method}")
            selected[track][method] = min(
                candidates,
                key=lambda row: (
                    -float(row["mean_psnr"]),
                    -float(row["mean_ssim"]),
                    float(row["mean_lpips"]),
                    float(row["median_ms"]),
                ),
            )
    return selected
