from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lowlight.classical.brighteners import GammaBrightener
from lowlight.classical.denoisers import GaussianDenoiser
from lowlight.models import build_model
from lowlight.pipelines import (
    BrightenThenDenoisePipeline,
    ClassicalBrightenerComponent,
    ClassicalDenoiserComponent,
    DenoiseThenBrightenPipeline,
    EndToEndPipeline,
    NeuralComponent,
    build_pipeline,
)
from lowlight.pipelines.factory import build_brightener_component, build_denoiser_component

ROOT = Path(__file__).resolve().parents[2]


def test_denoise_then_brighten_order_and_timing() -> None:
    pipeline = DenoiseThenBrightenPipeline(lambda x: x + 0.1, lambda x: x * 2.0)
    image = np.full((8, 8, 3), 0.2)
    output = pipeline.run_timed(image)
    assert np.allclose(output.image, 0.6)
    assert list(output.component_ms) == ["denoise", "brighten"]
    assert output.total_ms >= sum(output.component_ms.values())


def test_brighten_then_denoise_order() -> None:
    pipeline = BrightenThenDenoisePipeline(lambda x: x * 2.0, lambda x: x + 0.1)
    image = np.full((8, 8, 3), 0.2)
    assert np.allclose(pipeline.run(image), 0.5)


def test_input_is_not_mutated() -> None:
    pipeline = DenoiseThenBrightenPipeline(lambda x: x + 0.1, lambda x: x)
    image = np.full((8, 8, 3), 0.2)
    snapshot = image.copy()
    pipeline.run(image)
    assert np.array_equal(image, snapshot)


def test_pipeline_rejects_out_of_range_output() -> None:
    pipeline = DenoiseThenBrightenPipeline(lambda x: x, lambda x: x + 2.0)
    with pytest.raises(ValueError, match="outside"):
        pipeline.run(np.full((8, 8, 3), 0.2))


def test_pipeline_rejects_shape_change() -> None:
    pipeline = DenoiseThenBrightenPipeline(lambda x: x, lambda x: x[:-1])
    with pytest.raises(ValueError, match="changed shape"):
        pipeline.run(np.full((8, 8, 3), 0.2))


def test_classical_components_are_deterministic() -> None:
    denoiser = ClassicalDenoiserComponent(GaussianDenoiser(sigma=0.8))
    brightener = ClassicalBrightenerComponent(GammaBrightener(gamma=0.5))
    pipeline = DenoiseThenBrightenPipeline(denoiser, brightener)
    image = np.random.default_rng(0).random((16, 16, 3))
    first = pipeline.run(image)
    second = pipeline.run(image)
    assert np.array_equal(first, second)
    metadata = pipeline.metadata()
    assert metadata["steps"][0]["component"]["name"] == "gaussian"
    assert metadata["steps"][1]["component"]["name"] == "gamma"


def test_neural_component_cpu() -> None:
    model = build_model({"base_channels": 4, "depth": 2, "groups": 2, "residual": False})
    component = NeuralComponent(model, device="cpu", name="tiny")
    image = np.random.default_rng(1).random((16, 16, 3))
    output = component(image)
    assert output.shape == image.shape
    assert output.min() >= 0.0
    assert output.max() <= 1.0
    assert component.metadata()["parameters"] > 0


def test_end_to_end_wraps_one_step() -> None:
    pipeline = EndToEndPipeline(lambda x: np.clip(x * 3.0, 0.0, 1.0))
    output = pipeline.run_timed(np.full((8, 8, 3), 0.2))
    assert np.allclose(output.image, 0.6)
    assert list(output.component_ms) == ["end_to_end"]


def test_build_components_from_config(tmp_path: Path) -> None:
    denoiser_config = tmp_path / "gaussian.yaml"
    denoiser_config.write_text(
        "name: gaussian\nparams:\n  sigma: 0.5\n",
        encoding="utf-8",
    )
    denoiser = build_denoiser_component(
        {"kind": "classical", "config": str(denoiser_config)}, root=tmp_path
    )
    brightener = build_brightener_component(
        {
            "kind": "classical",
            "config": "configs/classical/brightening_frozen.yaml",
            "track": "raw",
            "method": "gamma",
        },
        root=ROOT,
    )
    assert denoiser.metadata()["name"] == "gaussian"
    assert brightener.metadata()["name"] == "gamma"


def test_build_pipeline_classical_classical() -> None:
    config = {
        "order": "denoise_then_brighten",
        "denoiser": {
            "kind": "classical",
            "name": "gaussian",
            "params": {"sigma": 0.5},
        },
        "brightener": {
            "kind": "classical",
            "config": "configs/classical/brightening_frozen.yaml",
            "track": "raw",
            "method": "gamma",
        },
    }
    pipeline = build_pipeline(config, root=ROOT)
    assert pipeline.name == "denoise_then_brighten"
    assert pipeline.metadata()["steps"][0]["name"] == "denoise"


def test_build_pipeline_rejects_missing_checkpoint() -> None:
    config = {
        "order": "end_to_end",
        "model": {"kind": "cnn", "checkpoint_root": "missing/end_to_end"},
    }
    with pytest.raises(FileNotFoundError, match="missing checkpoint"):
        build_pipeline(config, root=ROOT)
