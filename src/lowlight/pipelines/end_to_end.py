"""Single-model end-to-end pipeline."""

from __future__ import annotations

from collections.abc import Callable

from lowlight.pipelines.base import BasePipeline


class EndToEndPipeline(BasePipeline):
    name = "end_to_end"

    def __init__(self, model: Callable) -> None:
        super().__init__([("end_to_end", model)])
