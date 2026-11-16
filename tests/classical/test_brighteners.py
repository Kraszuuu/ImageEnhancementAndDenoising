from __future__ import annotations

import numpy as np
import pytest

from lowlight.classical.brighteners import (
    ClaheBrightener,
    GammaBrightener,
    MsrcrBrightener,
    build_brightener,
)
from lowlight.metrics.base import MetricInputError

BRIGHTENERS = [
    GammaBrightener(gamma=0.4),
    ClaheBrightener(clip_limit=0.01, tile_grid_size=8),
    MsrcrBrightener(scales=(15.0, 80.0, 250.0), gain=2.0, offset=0.25),
]


def _image(seed: int = 0, shape: tuple[int, ...] = (48, 48, 3)) -> np.ndarray:
    return np.random.default_rng(seed).random(shape)


@pytest.mark.parametrize("brightener", BRIGHTENERS, ids=lambda item: item.name)
def test_rgb_output_contract(brightener) -> None:
    image = _image()
    original = image.copy()
    output = brightener.enhance(image)
    assert output.shape == image.shape
    assert output.dtype == np.float64
    assert output.min() >= 0.0
    assert output.max() <= 1.0
    assert np.array_equal(image, original)


@pytest.mark.parametrize("brightener", BRIGHTENERS, ids=lambda item: item.name)
def test_grayscale_is_supported(brightener) -> None:
    image = _image(1, (48, 48))
    output = brightener.enhance(image)
    assert output.shape == image.shape
    assert np.all(np.isfinite(output))


@pytest.mark.parametrize("brightener", BRIGHTENERS, ids=lambda item: item.name)
def test_single_channel_is_supported(brightener) -> None:
    image = _image(2, (48, 48, 1))
    output = brightener.enhance(image)
    assert output.shape == image.shape


@pytest.mark.parametrize("brightener", BRIGHTENERS, ids=lambda item: item.name)
def test_deterministic(brightener) -> None:
    image = _image(3)
    assert np.array_equal(brightener.enhance(image), brightener.enhance(image))


def test_gamma_one_is_identity() -> None:
    image = _image(4)
    assert np.array_equal(GammaBrightener(gamma=1.0).enhance(image), image)


def test_clahe_small_image() -> None:
    image = _image(5, (16, 16, 3))
    output = ClaheBrightener(clip_limit=0.02, tile_grid_size=4).enhance(image)
    assert output.shape == image.shape
    assert np.all(np.isfinite(output))


@pytest.mark.parametrize("brightener", BRIGHTENERS, ids=lambda item: item.name)
def test_out_of_range_is_rejected(brightener) -> None:
    with pytest.raises(MetricInputError):
        brightener.enhance(np.full((16, 16, 3), 1.5))


def test_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        GammaBrightener(gamma=0.0)
    with pytest.raises(ValueError):
        ClaheBrightener(clip_limit=0.0)
    with pytest.raises(ValueError):
        ClaheBrightener(clip_limit=0.01, tile_grid_size=0)
    with pytest.raises(ValueError):
        MsrcrBrightener(scales=())
    with pytest.raises(ValueError):
        MsrcrBrightener(scales=(0.0, 80.0))
    with pytest.raises(ValueError):
        MsrcrBrightener(scales=(15.0,), gain=0.0)


def test_registry() -> None:
    brightener = build_brightener("gamma", {"gamma": 0.5})
    assert isinstance(brightener, GammaBrightener)
    with pytest.raises(ValueError):
        build_brightener("unknown", {})
