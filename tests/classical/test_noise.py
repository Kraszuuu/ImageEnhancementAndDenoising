from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lowlight.metrics.base import MetricInputError
from lowlight.utils.noise import (
    add_noise,
    linear_to_srgb,
    load_noise_config,
    srgb_to_linear,
    stable_seed,
)

ROOT = Path(__file__).resolve().parents[2]


def test_srgb_roundtrip() -> None:
    image = np.linspace(0.0, 1.0, 1024).reshape(32, 32)
    restored = linear_to_srgb(srgb_to_linear(image))
    assert np.allclose(restored, image, atol=1e-12)


def test_noise_is_deterministic_and_seed_sensitive() -> None:
    image = np.random.default_rng(0).random((32, 32, 3))
    first = add_noise(image, alpha=0.01, sigma=0.005, seed=7)
    second = add_noise(image, alpha=0.01, sigma=0.005, seed=7)
    other = add_noise(image, alpha=0.01, sigma=0.005, seed=8)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)


def test_clean_noise_is_identity_and_does_not_mutate() -> None:
    image = np.random.default_rng(1).random((16, 16, 3))
    original = image.copy()
    output = add_noise(image, alpha=0.0, sigma=0.0, seed=0)
    assert np.array_equal(output, image)
    assert np.array_equal(image, original)


def test_output_range_and_shape() -> None:
    image = np.random.default_rng(2).random((24, 32))
    output = add_noise(image, alpha=0.05, sigma=0.02, seed=0)
    assert output.shape == image.shape
    assert output.dtype == np.float64
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_invalid_noise_parameters_and_input() -> None:
    image = np.zeros((8, 8, 3))
    with pytest.raises(ValueError):
        add_noise(image, alpha=-1.0, sigma=0.0, seed=0)
    with pytest.raises(MetricInputError):
        add_noise(np.full((8, 8, 3), 2.0), alpha=0.01, sigma=0.0, seed=0)


def test_stable_seed_is_stable_and_distinct() -> None:
    assert stable_seed(0, "a.png", "low") == stable_seed(0, "a.png", "low")
    assert stable_seed(0, "a.png", "low") != stable_seed(0, "a.png", "medium")


def test_frozen_config_loads() -> None:
    config = load_noise_config(ROOT / "configs" / "data" / "noise.yaml")
    assert config.levels["clean"].alpha == 0.0
    assert config.levels["low"].alpha > 0.0
    assert config.levels["medium"].sigma >= config.levels["low"].sigma
    assert config.seed_policy == "fixed_per_image_and_level"


def test_config_add_level() -> None:
    config = load_noise_config(ROOT / "configs" / "data" / "noise.yaml")
    image = np.random.default_rng(3).random((16, 16, 3))
    assert np.array_equal(config.add(image, "clean", image_id="x"), image)
    noisy = config.add(image, "high", image_id="x")
    assert not np.array_equal(noisy, image)
