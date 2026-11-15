from __future__ import annotations

import numpy as np
import pytest

from lowlight.classical.denoisers import (
    BilateralDenoiser,
    Bm3dDenoiser,
    GaussianDenoiser,
    MedianDenoiser,
    NlMeansDenoiser,
    build_denoiser,
)
from lowlight.metrics.base import MetricInputError

DENOISERS = [
    GaussianDenoiser(sigma=1.0),
    MedianDenoiser(kernel_size=3),
    BilateralDenoiser(d=5, sigma_color=0.1, sigma_space=2.0),
    NlMeansDenoiser(h=0.5, patch_size=5, patch_distance=5),
    Bm3dDenoiser(sigma_psd=0.05),
]


def _image(seed: int = 0, shape: tuple[int, ...] = (48, 48, 3)) -> np.ndarray:
    return np.random.default_rng(seed).random(shape)


@pytest.mark.parametrize("denoiser", DENOISERS, ids=lambda item: item.name)
def test_rgb_output_contract(denoiser) -> None:
    image = _image()
    original = image.copy()
    output = denoiser.denoise(image)
    assert output.shape == image.shape
    assert output.dtype == np.float64
    assert output.min() >= 0.0
    assert output.max() <= 1.0
    assert np.array_equal(image, original)


@pytest.mark.parametrize("denoiser", DENOISERS, ids=lambda item: item.name)
def test_grayscale_is_supported(denoiser) -> None:
    image = _image(1, (48, 48))
    output = denoiser.denoise(image)
    assert output.shape == image.shape
    assert np.all(np.isfinite(output))


@pytest.mark.parametrize("denoiser", DENOISERS, ids=lambda item: item.name)
def test_single_channel_is_supported(denoiser) -> None:
    image = _image(2, (48, 48, 1))
    output = denoiser.denoise(image)
    assert output.shape == image.shape


@pytest.mark.parametrize("denoiser", DENOISERS, ids=lambda item: item.name)
def test_deterministic(denoiser) -> None:
    image = _image(3)
    first = denoiser.denoise(image)
    second = denoiser.denoise(image)
    # BM3D parallel FFT may differ at ~1e-7 even with identical input.
    assert np.allclose(first, second, atol=1e-6)


@pytest.mark.parametrize("denoiser", DENOISERS, ids=lambda item: item.name)
def test_out_of_range_is_rejected(denoiser) -> None:
    with pytest.raises(MetricInputError):
        denoiser.denoise(np.full((16, 16, 3), 1.5))


def test_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        GaussianDenoiser(sigma=0.0)
    with pytest.raises(ValueError):
        MedianDenoiser(kernel_size=4)
    with pytest.raises(ValueError):
        BilateralDenoiser(d=4, sigma_color=0.1, sigma_space=2.0)
    with pytest.raises(ValueError):
        NlMeansDenoiser(h=0.0)
    with pytest.raises(ValueError):
        Bm3dDenoiser(sigma_psd=-1.0)
    with pytest.raises(ValueError):
        Bm3dDenoiser(sigma_psd=0.05, num_threads=-1)


def test_registry() -> None:
    denoiser = build_denoiser("gaussian", {"sigma": 0.7})
    assert isinstance(denoiser, GaussianDenoiser)
    with pytest.raises(ValueError):
        build_denoiser("unknown", {})
