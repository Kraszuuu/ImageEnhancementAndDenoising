from __future__ import annotations

import numpy as np

from lowlight.classical.brighteners import GammaBrightener
from lowlight.classical.brightening_pipeline import ClassicalBrighteningPipeline
from lowlight.classical.denoisers import GaussianDenoiser


def test_pipeline_without_denoiser() -> None:
    image = np.random.default_rng(0).random((32, 32, 3))
    original = image.copy()
    pipeline = ClassicalBrighteningPipeline(brightener=GammaBrightener(gamma=0.5))
    brightened, source = pipeline.run(image)
    assert brightened.shape == image.shape
    assert source.shape == image.shape
    assert np.array_equal(source, image)
    assert not np.array_equal(brightened, image)
    assert np.array_equal(image, original)


def test_pipeline_with_denoiser() -> None:
    image = np.random.default_rng(1).random((32, 32, 3))
    pipeline = ClassicalBrighteningPipeline(
        brightener=GammaBrightener(gamma=0.5),
        denoiser=GaussianDenoiser(sigma=1.0),
    )
    brightened, source = pipeline.run(image)
    assert brightened.shape == image.shape
    assert source.shape == image.shape
    assert not np.array_equal(source, image)
    assert not np.array_equal(brightened, source)


def test_pipeline_metadata() -> None:
    pipeline = ClassicalBrighteningPipeline(brightener=GammaBrightener(gamma=0.5))
    metadata = pipeline.metadata()
    assert metadata["brightener"]["name"] == "gamma"
    assert metadata["denoiser"] is None
