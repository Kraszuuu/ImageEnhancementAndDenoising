"""Shared, framework-independent inference helpers for the Streamlit demo."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from lowlight.metrics import NiqeMetric, as_image, resolve_device
from lowlight.pipelines import build_pipeline
from lowlight.pipelines.base import BasePipeline, PipelineOutput
from lowlight.pipelines.factory import (
    build_brightener_component,
    build_denoiser_component,
)
from lowlight.utils.io import load_rgb_bytes

ROOT = Path(__file__).resolve().parents[3]
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_SIDE = 2048
MAX_IMAGE_PIXELS = 4_000_000
MIN_IMAGE_SIDE = 32
SUPPORTED_FORMATS = {"PNG", "JPEG"}
SUPPORTED_SEEDS = (0, 1, 2)
NIQE_MIN_SIDE = 96


class DemoError(ValueError):
    """Base class for errors that can be shown to a demo user."""


class ImageValidationError(DemoError):
    """Raised when an uploaded file is not a supported, usable image."""


class MissingCheckpointError(DemoError):
    """Raised when a selected model has no local checkpoint."""


class NiqeUnavailableError(DemoError):
    """Raised when NIQE cannot be computed for an image."""


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    group: str
    description: str
    config: str | None = None
    variant: str | None = None
    component: str | None = None
    checkpoint_root: str | None = None
    model_config: str | None = None
    seedable: bool = True
    stage: str = "pipeline"


METHOD_SPECS: dict[str, MethodSpec] = {
    "classical_classical": MethodSpec(
        key="classical_classical",
        label="Classical pipeline (BM3D -> gamma)",
        group="Classical",
        description="Frozen BM3D denoiser followed by frozen gamma brightening.",
        config="configs/pipelines/classical_classical.yaml",
        seedable=False,
        stage="Stage 06",
    ),
    "brighten_then_denoise": MethodSpec(
        key="brighten_then_denoise",
        label="Classical pipeline (gamma -> BM3D)",
        group="Classical",
        description="Frozen gamma brightening followed by frozen BM3D denoising.",
        config="configs/pipelines/brighten_then_denoise.yaml",
        seedable=False,
        stage="Stage 06",
    ),
    "cnn_denoise": MethodSpec(
        key="cnn_denoise",
        label="U-Net denoiser",
        group="U-Net",
        description="Residual U-Net trained to remove calibrated low-light noise.",
        component="denoiser",
        checkpoint_root="checkpoints/denoising_cnn",
        model_config="configs/models/unet_denoising.yaml",
        stage="Stage 03",
    ),
    "cnn_brighten_raw": MethodSpec(
        key="cnn_brighten_raw",
        label="U-Net brightener (raw input)",
        group="U-Net",
        description="Direct-output U-Net brightener trained on raw low-light images.",
        component="brightener",
        checkpoint_root="checkpoints/brightening_cnn/raw",
        model_config="configs/models/unet_brightening.yaml",
        stage="Stage 05",
    ),
    "classical_cnn": MethodSpec(
        key="classical_cnn",
        label="Classical denoiser -> U-Net brightener",
        group="Pipelines",
        description="BM3D denoising followed by a U-Net brightener trained on BM3D outputs.",
        config="configs/pipelines/classical_cnn.yaml",
        stage="Stage 06",
    ),
    "cnn_classical": MethodSpec(
        key="cnn_classical",
        label="U-Net denoiser -> classical brightener",
        group="Pipelines",
        description="U-Net denoising followed by frozen gamma brightening.",
        config="configs/pipelines/cnn_classical.yaml",
        stage="Stage 06",
    ),
    "cnn_cnn": MethodSpec(
        key="cnn_cnn",
        label="U-Net denoiser -> U-Net brightener",
        group="Pipelines",
        description="U-Net denoising followed by a U-Net brightener trained on U-Net outputs.",
        config="configs/pipelines/cnn_cnn.yaml",
        stage="Stage 06",
    ),
    "end_to_end_real": MethodSpec(
        key="end_to_end_real",
        label="End-to-end U-Net (real input)",
        group="U-Net",
        description="Single U-Net trained directly on paired real low-light images.",
        config="configs/pipelines/end_to_end.yaml",
        variant="real",
        stage="Stage 06",
    ),
    "end_to_end_controlled": MethodSpec(
        key="end_to_end_controlled",
        label="End-to-end U-Net (controlled noise)",
        group="U-Net",
        description="Single U-Net trained on clean images with calibrated synthetic noise.",
        config="configs/pipelines/end_to_end.yaml",
        variant="controlled",
        stage="Stage 06",
    ),
}


@dataclass(frozen=True)
class UploadedImage:
    image: np.ndarray
    width: int
    height: int
    format: str
    source_mode: str


@dataclass(frozen=True)
class InferenceResult:
    image: np.ndarray
    method: MethodSpec
    seed: int
    device: str
    total_ms: float
    component_ms: dict[str, float]
    pipeline_metadata: dict[str, object]


@dataclass(frozen=True)
class DemoEngine:
    method: MethodSpec
    seed: int
    device: str
    pipeline: BasePipeline


class _SingleComponentPipeline(BasePipeline):
    def __init__(
        self, name: str, step_name: str, component: Callable[[np.ndarray], np.ndarray]
    ) -> None:
        super().__init__([(step_name, component)])
        self.name = name


def method_keys() -> list[str]:
    return list(METHOD_SPECS)


def method_spec(key: str) -> MethodSpec:
    try:
        return METHOD_SPECS[key]
    except KeyError as error:
        raise DemoError(f"unknown method: {key!r}") from error


def load_uploaded_image(
    data: bytes,
    *,
    filename: str | None = None,
    max_bytes: int = MAX_UPLOAD_BYTES,
    max_side: int = MAX_IMAGE_SIDE,
    max_pixels: int = MAX_IMAGE_PIXELS,
    min_side: int = MIN_IMAGE_SIDE,
) -> UploadedImage:
    """Validate an uploaded PNG/JPEG and convert it exactly like evaluation."""
    name = filename or "uploaded image"
    if not data:
        raise ImageValidationError(f"{name}: file is empty")
    if len(data) > max_bytes:
        raise ImageValidationError(
            f"{name}: file is {len(data) / (1024 * 1024):.1f} MB; "
            f"limit is {max_bytes / (1024 * 1024):.0f} MB"
        )
    try:
        with Image.open(BytesIO(data)) as handle:
            image_format = (handle.format or "").upper()
            if image_format not in SUPPORTED_FORMATS:
                supported = ", ".join(sorted(SUPPORTED_FORMATS))
                raise ImageValidationError(
                    f"{name}: unsupported format {image_format or 'unknown'}; "
                    f"use {supported}"
                )
            width, height = handle.size
            source_mode = handle.mode
            if width < min_side or height < min_side:
                raise ImageValidationError(
                    f"{name}: image is {width}x{height}; minimum side is {min_side} px"
                )
            if width > max_side or height > max_side:
                raise ImageValidationError(
                    f"{name}: image is {width}x{height}; maximum side is {max_side} px"
                )
            if width * height > max_pixels:
                raise ImageValidationError(
                    f"{name}: image is {width * height:,} pixels; "
                    f"limit is {max_pixels:,}"
                )
            array = load_rgb_bytes(data)
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise ImageValidationError(f"{name}: cannot decode image ({error})") from error
    return UploadedImage(
        image=array,
        width=width,
        height=height,
        format=image_format,
        source_mode=source_mode,
    )


def load_engine(
    method_key: str,
    *,
    seed: int = 0,
    device: str = "auto",
    root: Path = ROOT,
) -> DemoEngine:
    spec = method_spec(method_key)
    if spec.seedable:
        if seed not in SUPPORTED_SEEDS:
            raise DemoError(f"seed must be one of {SUPPORTED_SEEDS}, got {seed}")
    else:
        seed = 0
    resolved_device = str(resolve_device(None if device == "auto" else device))
    try:
        if spec.config is not None:
            pipeline = build_pipeline(
                spec.config,
                root=root,
                seed=seed,
                device=resolved_device,
                variant=spec.variant,
            )
        elif spec.component == "denoiser":
            denoiser_component = build_denoiser_component(
                {
                    "kind": "cnn",
                    "checkpoint_root": spec.checkpoint_root,
                    "model_config": spec.model_config,
                },
                root=root,
                seed=seed,
                device=resolved_device,
            )
            pipeline = _SingleComponentPipeline(
                "cnn_denoise", "denoise", denoiser_component
            )
        elif spec.component == "brightener":
            brightener_component = build_brightener_component(
                {
                    "kind": "cnn",
                    "checkpoint_root": spec.checkpoint_root,
                    "model_config": spec.model_config,
                },
                root=root,
                seed=seed,
                device=resolved_device,
            )
            pipeline = _SingleComponentPipeline(
                "cnn_brighten", "brighten", brightener_component
            )
        else:
            raise DemoError(f"method {spec.key!r} has no loader")
    except FileNotFoundError as error:
        raise MissingCheckpointError(
            f"{error}. Retrain the model from README.md or choose a method whose "
            "checkpoint exists locally."
        ) from error
    return DemoEngine(
        method=spec,
        seed=seed,
        device=resolved_device,
        pipeline=pipeline,
    )


def run_engine(engine: DemoEngine, image: object) -> InferenceResult:
    array = as_image(image, name="uploaded image")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ImageValidationError(
            f"expected an RGB image after preprocessing, got shape {array.shape}"
        )
    output: PipelineOutput = engine.pipeline.run_timed(array)
    return InferenceResult(
        image=output.image,
        method=engine.method,
        seed=engine.seed,
        device=engine.device,
        total_ms=output.total_ms,
        component_ms=dict(output.component_ms),
        pipeline_metadata=engine.pipeline.metadata(),
    )


def run_inference(
    image: object,
    method_key: str,
    *,
    seed: int = 0,
    device: str = "auto",
    root: Path = ROOT,
) -> InferenceResult:
    return run_engine(load_engine(method_key, seed=seed, device=device, root=root), image)


def load_niqe(device: str = "auto") -> NiqeMetric:
    return NiqeMetric(device=None if device == "auto" else device)


def score_niqe(image: object, *, metric: NiqeMetric | None = None) -> float:
    array = as_image(image, name="niqe image")
    height, width = array.shape[:2]
    if height < NIQE_MIN_SIDE or width < NIQE_MIN_SIDE:
        raise NiqeUnavailableError(
            f"NIQE requires at least {NIQE_MIN_SIDE} px per side; got {width}x{height}"
        )
    try:
        scorer = metric or load_niqe()
        return scorer.compute(array, None)
    except NiqeUnavailableError:
        raise
    except Exception as error:
        raise NiqeUnavailableError(str(error)) from error


def image_to_png_bytes(image: object) -> bytes:
    array = as_image(image, name="output image")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ImageValidationError(f"expected RGB output, got shape {array.shape}")
    encoded = (np.clip(array, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(encoded, mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


__all__ = [
    "DemoEngine",
    "DemoError",
    "ImageValidationError",
    "InferenceResult",
    "MAX_IMAGE_PIXELS",
    "MAX_IMAGE_SIDE",
    "MAX_UPLOAD_BYTES",
    "METHOD_SPECS",
    "MIN_IMAGE_SIDE",
    "MethodSpec",
    "MissingCheckpointError",
    "NIQE_MIN_SIDE",
    "NiqeUnavailableError",
    "SUPPORTED_SEEDS",
    "UploadedImage",
    "image_to_png_bytes",
    "load_engine",
    "load_niqe",
    "load_uploaded_image",
    "method_keys",
    "method_spec",
    "run_engine",
    "run_inference",
    "score_niqe",
]
