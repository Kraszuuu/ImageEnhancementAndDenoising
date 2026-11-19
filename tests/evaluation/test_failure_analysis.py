from __future__ import annotations

import numpy as np
import pytest

from lowlight.evaluation.failure_analysis import analyze_rows, failure_metrics


def test_identical_images_have_zero_error() -> None:
    image = np.full((8, 8, 3), 0.25, dtype=np.float64)
    metrics = failure_metrics(image, image)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["brightness_error"] == pytest.approx(0.0)
    assert metrics["color_cast"] == pytest.approx(0.0)
    assert metrics["detail_loss"] == pytest.approx(0.0)


def test_clipping_and_color_cast_are_measured() -> None:
    target = np.full((4, 4, 3), 0.5, dtype=np.float64)
    prediction = target.copy()
    prediction[..., 0] = 1.0
    prediction[..., 1] = 0.0
    metrics = failure_metrics(prediction, target)
    assert metrics["color_cast"] == pytest.approx(0.5)
    assert metrics["clip_high_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["clip_low_fraction"] == pytest.approx(1.0 / 3.0)


def test_analyze_rows_selects_worst() -> None:
    rows = [{"image_id": "a", "psnr": "20"}, {"image_id": "b", "psnr": "10"}]
    assert analyze_rows(rows, limit=1) == [{"image_id": "b", "psnr": "10"}]
    assert analyze_rows(rows, worst=False, limit=1) == [{"image_id": "a", "psnr": "20"}]
