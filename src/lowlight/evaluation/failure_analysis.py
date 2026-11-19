"""Failure-oriented image diagnostics for Stage 07 case studies."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lowlight.metrics.base import as_image_pair


def _gradient_magnitude(image: np.ndarray) -> float:
    gray = image.mean(axis=2) if image.ndim == 3 else image
    dx = np.abs(np.diff(gray, axis=1)).mean()
    dy = np.abs(np.diff(gray, axis=0)).mean()
    return float(dx + dy) / 2.0


def failure_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    source: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute interpretable failure diagnostics in ``[0, 1]`` image space."""
    prediction, target = as_image_pair(prediction, target)
    residual = prediction - target

    channel_shift = np.abs(prediction.mean(axis=(0, 1)) - target.mean(axis=(0, 1)))
    target_gradient = _gradient_magnitude(target)
    prediction_gradient = _gradient_magnitude(prediction)
    if target_gradient <= 1e-12 and prediction_gradient <= 1e-12:
        detail_ratio = 1.0
    else:
        detail_ratio = prediction_gradient / max(target_gradient, 1e-12)
    metrics = {
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "brightness_error": float(abs(prediction.mean() - target.mean())),
        "color_cast": float(channel_shift.max()),
        "clip_low_fraction": float((prediction <= 1e-6).mean()),
        "clip_high_fraction": float((prediction >= 1.0 - 1e-6).mean()),
        "residual_std": float(residual.std()),
        "detail_ratio": float(detail_ratio),
        "detail_loss": float(1.0 - detail_ratio),
    }
    if source is not None:
        source, _ = as_image_pair(source, target)
        source_residual = float(np.std(source - target))
        metrics["noise_amplification"] = float(
            metrics["residual_std"] / max(source_residual, 1e-12)
        )
    return metrics


def analyze_rows(
    rows: Sequence[dict[str, object]],
    *,
    metric: str = "psnr",
    worst: bool = True,
    limit: int | None = None,
) -> list[dict[str, object]]:
    """Select the worst or best per-image rows for a quality metric."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    usable = []
    for row in rows:
        value = row.get(metric, "")
        try:
            parsed = float(str(value))
        except TypeError, ValueError:
            continue
        if np.isfinite(parsed):
            usable.append((parsed, row))
    usable.sort(key=lambda item: item[0], reverse=not worst)
    selected = usable if limit is None else usable[:limit]
    return [dict(row) for _, row in selected]
