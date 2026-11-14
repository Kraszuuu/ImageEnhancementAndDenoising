"""Inference timing protocol: batch 1, warm-up, repeated forward passes."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


@dataclass(frozen=True)
class TimingResult:
    device: str
    mean_ms: float
    std_ms: float
    median_ms: float
    n: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class InferenceTimer:
    """Measure per-forward latency with explicit CUDA synchronization."""

    def __init__(self, warmup: int = 10, iterations: int = 50) -> None:
        if warmup < 0:
            raise ValueError("warmup must be non-negative")
        if iterations < 1:
            raise ValueError("iterations must be positive")
        self.warmup = warmup
        self.iterations = iterations

    def measure(
        self, forward: Callable[[], object], device: str = "cpu"
    ) -> TimingResult:
        import numpy as np

        for _ in range(self.warmup):
            forward()
        _synchronize(device)

        samples = []
        for _ in range(self.iterations):
            _synchronize(device)
            start = time.perf_counter()
            forward()
            _synchronize(device)
            samples.append((time.perf_counter() - start) * 1000.0)

        values = np.asarray(samples, dtype=np.float64)
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        return TimingResult(
            device=device,
            mean_ms=float(values.mean()),
            std_ms=std,
            median_ms=float(np.median(values)),
            n=int(values.size),
        )
