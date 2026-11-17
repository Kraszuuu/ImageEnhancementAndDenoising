"""Brighten then denoise ablation pipeline."""

from __future__ import annotations

from collections.abc import Callable

from lowlight.pipelines.base import BasePipeline


class BrightenThenDenoisePipeline(BasePipeline):
    name = "brighten_then_denoise"

    def __init__(
        self,
        brightener: Callable,
        denoiser: Callable,
    ) -> None:
        super().__init__([("brighten", brightener), ("denoise", denoiser)])
