from __future__ import annotations

import math

import numpy as np
import pytest

from lowlight.metrics.base import MetricInputError
from lowlight.metrics.psnr import PsnrMetric, mse, psnr_from_mse, to_luminance


def _images(seed: int = 0, shape: tuple[int, ...] = (32, 48, 3)):
    rng = np.random.default_rng(seed)
    target = rng.random(shape)
    prediction = np.clip(0.9 * target + 0.05 * rng.random(shape), 0.0, 1.0)
    return prediction, target


def test_identical_images_give_infinity() -> None:
    _, target = _images()
    assert PsnrMetric().compute(target, target) == math.inf


def test_constant_black_and_white_is_zero_db() -> None:
    black = np.zeros((8, 8, 3))
    white = np.ones((8, 8, 3))
    assert PsnrMetric().compute(black, white) == pytest.approx(0.0)


def test_rgb_matches_numpy_reference() -> None:
    prediction, target = _images()
    reference = 10.0 * math.log10(1.0 / np.mean((prediction - target) ** 2))
    assert PsnrMetric().compute(prediction, target) == pytest.approx(reference, abs=1e-6)


def test_luma_matches_reference() -> None:
    prediction, target = _images(1)
    pred_luma = to_luminance(prediction)
    ref_luma = to_luminance(target)
    reference = 10.0 * math.log10(1.0 / np.mean((pred_luma - ref_luma) ** 2))
    metric = PsnrMetric(channel="luma")
    assert metric.compute(prediction, target) == pytest.approx(reference, abs=1e-6)


def test_grayscale_input_is_supported() -> None:
    prediction, target = _images(2, (16, 16))
    assert math.isfinite(PsnrMetric().compute(prediction, target))


def test_out_of_range_and_nan_are_rejected() -> None:
    target = np.zeros((8, 8, 3))
    with pytest.raises(MetricInputError):
        PsnrMetric().compute(np.full((8, 8, 3), 1.5), target)
    with pytest.raises(MetricInputError):
        PsnrMetric().compute(np.full((8, 8, 3), np.nan), target)


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(MetricInputError):
        PsnrMetric().compute(np.zeros((8, 8, 3)), np.zeros((9, 8, 3)))


def test_invalid_channel_and_negative_mse() -> None:
    with pytest.raises(ValueError):
        PsnrMetric(channel="yuv")
    with pytest.raises(ValueError):
        psnr_from_mse(-1.0)


def test_mse_helper() -> None:
    assert mse(np.ones((4, 4)), np.zeros((4, 4))) == pytest.approx(1.0)
