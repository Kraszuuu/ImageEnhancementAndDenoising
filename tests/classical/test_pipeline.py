from __future__ import annotations

from pathlib import Path

import numpy as np

from lowlight.classical.denoisers import GaussianDenoiser
from lowlight.classical.pipeline import ClassicalDenoisingPipeline
from lowlight.utils.noise import load_noise_config

ROOT = Path(__file__).resolve().parents[2]


def test_pipeline_adds_noise_and_denoises() -> None:
    config = load_noise_config(ROOT / "configs" / "data" / "noise.yaml")
    pipeline = ClassicalDenoisingPipeline(
        denoiser=GaussianDenoiser(sigma=0.8), noise_config=config, seed=0
    )
    image = np.random.default_rng(0).random((32, 32, 3))
    denoised, noisy = pipeline.run(image, "medium", image_id="sample.png")
    assert denoised.shape == image.shape
    assert noisy.shape == image.shape
    assert not np.array_equal(noisy, image)
    assert not np.array_equal(denoised, noisy)
    assert np.array_equal(image, np.random.default_rng(0).random((32, 32, 3)))


def test_pipeline_is_deterministic() -> None:
    config = load_noise_config(ROOT / "configs" / "data" / "noise.yaml")
    pipeline = ClassicalDenoisingPipeline(
        denoiser=GaussianDenoiser(sigma=0.8), noise_config=config, seed=3
    )
    image = np.random.default_rng(1).random((24, 24, 3))
    first = pipeline.run(image, "low", image_id="a.png")
    second = pipeline.run(image, "low", image_id="a.png")
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_metadata() -> None:
    config = load_noise_config(ROOT / "configs" / "data" / "noise.yaml")
    pipeline = ClassicalDenoisingPipeline(
        denoiser=GaussianDenoiser(sigma=0.8), noise_config=config
    )
    metadata = pipeline.metadata()
    assert metadata["denoiser"]["name"] == "gaussian"
    assert set(metadata["noise_levels"]) == {"clean", "low", "medium", "high"}
