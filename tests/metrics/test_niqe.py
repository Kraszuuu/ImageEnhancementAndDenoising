from __future__ import annotations

import numpy as np
import pytest

from lowlight.metrics.base import MetricInputError
from lowlight.metrics.niqe import NiqeMetric


@pytest.fixture(scope="module")
def metric() -> NiqeMetric:
    return NiqeMetric(device="cpu")


def test_score_is_finite_and_positive(metric: NiqeMetric) -> None:
    image = np.random.default_rng(0).random((128, 128, 3))
    score = metric.compute(image, None)
    assert np.isfinite(score)
    assert score > 0.0


def test_deterministic(metric: NiqeMetric) -> None:
    image = np.random.default_rng(1).random((128, 128, 3))
    assert metric.compute(image, None) == pytest.approx(
        metric.compute(image, None), abs=1e-6
    )


def test_small_image_is_rejected(metric: NiqeMetric) -> None:
    image = np.random.default_rng(2).random((64, 64, 3))
    with pytest.raises(MetricInputError):
        metric.compute(image, None)


def test_metadata_records_params(metric: NiqeMetric) -> None:
    metadata = metric.metadata()
    assert metadata["package"] == "pyiqa"
    assert metadata["reference_required"] is False
