from __future__ import annotations

import numpy as np
import pytest

from lowlight.metrics.base import MetricInputError
from lowlight.metrics.lpips_metric import LpipsMetric


@pytest.fixture(scope="module")
def metric() -> LpipsMetric:
    return LpipsMetric(device="cpu")


def _image(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random((64, 64, 3))


def test_identical_images_give_zero(metric: LpipsMetric) -> None:
    image = _image(0)
    assert metric.compute(image, image) == pytest.approx(0.0, abs=1e-6)


def test_different_images_give_positive_distance(metric: LpipsMetric) -> None:
    assert metric.compute(_image(1), _image(2)) > 0.0


def test_deterministic(metric: LpipsMetric) -> None:
    prediction, target = _image(3), _image(4)
    first = metric.compute(prediction, target)
    second = metric.compute(prediction, target)
    assert first == pytest.approx(second, abs=1e-6)


def test_metadata_records_weights(metric: LpipsMetric) -> None:
    metadata = metric.metadata()
    assert metadata["package"] == "lpips"
    assert metadata["backbone"] == "alex"
    assert metadata["weights_sha256"]


def test_out_of_range_is_rejected(metric: LpipsMetric) -> None:
    target = _image(5)
    with pytest.raises(MetricInputError):
        metric.compute(target * 2.0, target)


def test_grayscale_is_rejected(metric: LpipsMetric) -> None:
    image = _image(6)[..., 0]
    with pytest.raises(MetricInputError):
        metric.compute(image, image)
