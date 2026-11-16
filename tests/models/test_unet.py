from __future__ import annotations

import pytest
import torch

from lowlight.models import (
    CompactUNet,
    UNetConfig,
    build_model,
    count_parameters,
    denoise_full_image,
    pad_to_multiple,
    unpad,
)


def _model(**overrides) -> CompactUNet:
    config = UNetConfig(base_channels=8, depth=3, **overrides)
    return CompactUNet(config)


def test_forward_shape_dtype_and_range() -> None:
    model = _model()
    inputs = torch.rand(2, 3, 32, 32)
    output = model(inputs)
    assert output.shape == inputs.shape
    assert output.dtype == torch.float32
    assert output.min() >= 0.0
    assert output.max() <= 1.0


def test_sigmoid_output_activation() -> None:
    model = _model(output_activation="sigmoid")
    output = model(torch.rand(1, 3, 32, 32))
    assert output.min() > 0.0
    assert output.max() < 1.0


def test_residual_model_starts_at_identity() -> None:
    model = _model(residual=True)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
    inputs = torch.rand(1, 3, 32, 32)
    assert torch.equal(model(inputs), inputs)


def test_backward_has_gradients() -> None:
    model = _model()
    inputs = torch.rand(1, 3, 32, 32, requires_grad=True)
    loss = torch.nn.functional.l1_loss(model(inputs), torch.zeros_like(inputs))
    loss.backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_reflection_padding_round_trip() -> None:
    tensor = torch.rand(1, 3, 37, 53)
    padded, padding = pad_to_multiple(tensor, 16)
    assert padded.shape[-2:] == (48, 64)
    restored = unpad(padded, padding)
    assert restored.shape == tensor.shape
    assert torch.equal(restored, tensor)


@pytest.mark.parametrize("shape", [(1, 3, 33, 47), (1, 3, 40, 40)])
def test_full_image_wrapper_restores_shape(shape) -> None:
    model = _model()
    output = denoise_full_image(model, torch.rand(*shape))
    assert output.shape == shape


def test_invalid_input_contract() -> None:
    model = _model()
    with pytest.raises(ValueError):
        model(torch.rand(1, 3, 30, 32))
    with pytest.raises(ValueError):
        model(torch.rand(1, 1, 32, 32))
    with pytest.raises(ValueError):
        model(torch.full((1, 3, 32, 32), float("nan")))
    with pytest.raises(ValueError):
        UNetConfig(depth=1)
    with pytest.raises(ValueError):
        UNetConfig(output_activation="tanh")


def test_build_model_from_config_and_parameter_count() -> None:
    model = build_model(
        {"base_channels": 8, "depth": 3, "groups": 8, "output_activation": "clamp"}
    )
    assert isinstance(model, CompactUNet)
    assert model.size_multiple == 8
    assert count_parameters(model) > 0
    assert model(torch.rand(1, 3, 24, 24)).shape == (1, 3, 24, 24)
