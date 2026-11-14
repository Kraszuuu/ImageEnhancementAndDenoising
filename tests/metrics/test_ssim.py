from __future__ import annotations

import numpy as np
import pytest
from skimage.metrics import structural_similarity

from lowlight.metrics.base import MetricInputError
from lowlight.metrics.ssim import SsimMetric, ssim


def _images(seed: int = 0, shape: tuple[int, ...] = (32, 48, 3)):
    rng = np.random.default_rng(seed)
    target = rng.random(shape)
    prediction = np.clip(0.85 * target + 0.1 * rng.random(shape), 0.0, 1.0)
    return prediction, target


def test_identical_images_give_one() -> None:
    _, target = _images()
    assert SsimMetric().compute(target, target) == pytest.approx(1.0, abs=1e-12)


def test_matches_frozen_skimage_reference() -> None:
    prediction, target = _images()
    reference = structural_similarity(
        target,
        prediction,
        channel_axis=2,
        data_range=1.0,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )
    assert SsimMetric().compute(prediction, target) == pytest.approx(reference, abs=1e-12)


def test_grayscale_input_is_supported() -> None:
    prediction, target = _images(1, (24, 24))
    reference = structural_similarity(
        target,
        prediction,
        channel_axis=None,
        data_range=1.0,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )
    assert ssim(prediction, target) == pytest.approx(reference, abs=1e-12)


def test_symmetry() -> None:
    prediction, target = _images(2)
    assert ssim(prediction, target) == pytest.approx(ssim(target, prediction), abs=1e-12)


def test_invalid_values_and_shapes_are_rejected() -> None:
    target = np.zeros((16, 16, 3))
    with pytest.raises(MetricInputError):
        SsimMetric().compute(np.full((16, 16, 3), -0.1), target)
    with pytest.raises(MetricInputError):
        SsimMetric().compute(np.zeros((16, 16, 3)), np.zeros((16, 17, 3)))
