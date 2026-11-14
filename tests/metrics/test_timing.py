from __future__ import annotations

import time

import pytest

from lowlight.metrics.timing import InferenceTimer


def test_timing_reports_expected_statistics() -> None:
    timer = InferenceTimer(warmup=1, iterations=5)
    result = timer.measure(lambda: time.sleep(0.001), device="cpu")
    assert result.device == "cpu"
    assert result.n == 5
    assert result.median_ms > 0.5
    assert result.mean_ms > 0.5
    assert result.std_ms >= 0.0


def test_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        InferenceTimer(warmup=-1)
    with pytest.raises(ValueError):
        InferenceTimer(iterations=0)


def test_cuda_timing_when_available() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    tensor = torch.rand(1, 3, 32, 32, device="cuda")
    model = torch.nn.Conv2d(3, 3, 3, padding=1).cuda().eval()
    timer = InferenceTimer(warmup=2, iterations=5)
    result = timer.measure(lambda: model(tensor), device="cuda")
    assert result.n == 5
    assert result.median_ms > 0.0
