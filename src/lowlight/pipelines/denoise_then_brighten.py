"""Denoise then brighten pipeline."""

from __future__ import annotations

from collections.abc import Callable

from lowlight.pipelines.base import BasePipeline


class DenoiseThenBrightenPipeline(BasePipeline):
    name = "denoise_then_brighten"

    def __init__(
        self,
        denoiser: Callable,
        brightener: Callable,
    ) -> None:
        super().__init__([("denoise", denoiser), ("brighten", brightener)])
