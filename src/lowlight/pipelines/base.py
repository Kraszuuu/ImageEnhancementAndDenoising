"""Shared interfaces for combined denoising and brightening pipelines."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from lowlight.metrics.base import as_image


@dataclass(frozen=True)
class PipelineOutput:
    image: np.ndarray
    component_ms: dict[str, float]
    total_ms: float


class BasePipeline:
    """Run ordered image components and validate their output contract."""

    name = "pipeline"

    def __init__(self, steps: Sequence[tuple[str, Callable[[np.ndarray], np.ndarray]]]) -> None:
        if not steps:
            raise ValueError("pipeline requires at least one step")
        names = [name for name, _ in steps]
        if len(names) != len(set(names)):
            raise ValueError("pipeline step names must be unique")
        self.steps = tuple(steps)

    def _run(self, image: object) -> PipelineOutput:
        current = as_image(image, name=f"{self.name} input").copy()
        expected_shape = current.shape
        component_ms: dict[str, float] = {}
        total_started = time.perf_counter()
        for name, component in self.steps:
            started = time.perf_counter()
            output = component(current)
            component_ms[name] = (time.perf_counter() - started) * 1000.0
            current = as_image(output, name=f"{self.name}:{name}")
            if current.shape != expected_shape:
                raise ValueError(
                    f"{self.name}:{name} changed shape {expected_shape} -> {current.shape}"
                )
        total_ms = (time.perf_counter() - total_started) * 1000.0
        return PipelineOutput(image=current, component_ms=component_ms, total_ms=total_ms)

    def run(self, image: object) -> np.ndarray:
        """Return the enhanced image; input is never mutated."""
        return self._run(image).image

    def run_timed(self, image: object) -> PipelineOutput:
        """Return the enhanced image and per-component wall-clock timings."""
        return self._run(image)

    def metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "steps": [
                {
                    "name": name,
                    "component": component.metadata()
                    if hasattr(component, "metadata")
                    else {"type": type(component).__name__},
                }
                for name, component in self.steps
            ],
        }
