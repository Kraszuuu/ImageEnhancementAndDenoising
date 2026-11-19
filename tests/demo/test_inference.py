from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from lowlight.demo.inference import (
    METHOD_SPECS,
    SUPPORTED_SEEDS,
    DemoEngine,
    DemoError,
    ImageValidationError,
    MissingCheckpointError,
    NiqeUnavailableError,
    image_to_png_bytes,
    load_engine,
    load_uploaded_image,
    method_keys,
    method_spec,
    run_engine,
    score_niqe,
)
from lowlight.pipelines.base import BasePipeline
from lowlight.utils.io import load_rgb


def _image_bytes(
    mode: str = "RGB",
    size: tuple[int, int] = (64, 48),
    image_format: str = "PNG",
) -> bytes:
    channels = 3 if mode == "RGB" else 1
    values = np.linspace(0, 255, size[0] * size[1] * channels, dtype=np.uint8)
    array = values.reshape((size[1], size[0], channels))
    if mode != "RGB":
        array = array[..., 0]
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format=image_format)
    return buffer.getvalue()


def test_method_registry_is_consistent() -> None:
    keys = method_keys()
    assert len(keys) == len(set(keys))
    assert set(keys) == set(METHOD_SPECS)
    for key in keys:
        assert method_spec(key).key == key


def test_supported_seeds_are_frozen() -> None:
    assert SUPPORTED_SEEDS == (0, 1, 2)


def test_png_bytes_match_path_loader(tmp_path: Path) -> None:
    data = _image_bytes()
    path = tmp_path / "sample.png"
    path.write_bytes(data)

    uploaded = load_uploaded_image(data, filename="sample.png")

    assert uploaded.width == 64
    assert uploaded.height == 48
    assert uploaded.format == "PNG"
    assert uploaded.source_mode == "RGB"
    np.testing.assert_allclose(uploaded.image, load_rgb(path))


def test_grayscale_is_promoted_to_rgb() -> None:
    uploaded = load_uploaded_image(_image_bytes(mode="L"), filename="gray.png")

    assert uploaded.source_mode == "L"
    assert uploaded.image.shape == (48, 64, 3)
    np.testing.assert_allclose(uploaded.image[..., 0], uploaded.image[..., 1])


def test_jpeg_is_supported() -> None:
    uploaded = load_uploaded_image(
        _image_bytes(image_format="JPEG"), filename="sample.jpg"
    )
    assert uploaded.format == "JPEG"
    assert uploaded.image.dtype == np.float64


def test_corrupt_file_is_rejected() -> None:
    with pytest.raises(ImageValidationError, match="cannot decode"):
        load_uploaded_image(b"not an image", filename="broken.png")


def test_empty_file_is_rejected() -> None:
    with pytest.raises(ImageValidationError, match="empty"):
        load_uploaded_image(b"", filename="empty.png")


def test_unsupported_format_is_rejected() -> None:
    buffer = BytesIO()
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(buffer, format="GIF")
    with pytest.raises(ImageValidationError, match="unsupported format"):
        load_uploaded_image(buffer.getvalue(), filename="sample.gif")


def test_too_small_image_is_rejected() -> None:
    data = _image_bytes(size=(16, 16))
    with pytest.raises(ImageValidationError, match="minimum side"):
        load_uploaded_image(data)


def test_large_image_limit_is_enforced() -> None:
    data = _image_bytes(size=(64, 64))
    with pytest.raises(ImageValidationError, match="maximum side"):
        load_uploaded_image(data, max_side=32)


def test_pixel_limit_is_enforced() -> None:
    data = _image_bytes(size=(64, 48))
    with pytest.raises(ImageValidationError, match="pixels"):
        load_uploaded_image(data, max_pixels=100)


def test_byte_limit_is_enforced() -> None:
    data = _image_bytes()
    with pytest.raises(ImageValidationError, match="limit is"):
        load_uploaded_image(data, max_bytes=10)


def test_missing_checkpoint_is_reported(tmp_path: Path) -> None:
    with pytest.raises(MissingCheckpointError, match="Retrain"):
        load_engine("cnn_denoise", root=tmp_path)


def test_invalid_seed_and_method_are_rejected() -> None:
    with pytest.raises(DemoError, match="seed"):
        load_engine("cnn_denoise", seed=9)
    with pytest.raises(DemoError, match="unknown method"):
        method_spec("nope")


class _IdentityComponent:
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return image.copy()

    def metadata(self) -> dict[str, object]:
        return {"kind": "identity"}


def test_run_engine_returns_timing_and_does_not_mutate() -> None:
    image = np.full((40, 40, 3), 0.25, dtype=np.float64)
    engine = DemoEngine(
        method=method_spec("classical_classical"),
        seed=0,
        device="cpu",
        pipeline=BasePipeline([("identity", _IdentityComponent())]),
    )

    result = run_engine(engine, image)

    np.testing.assert_allclose(result.image, image)
    np.testing.assert_allclose(image, 0.25)
    assert result.total_ms >= 0.0
    assert set(result.component_ms) == {"identity"}
    assert result.pipeline_metadata["name"] == "pipeline"


def test_niqe_rejects_small_images_before_loading_model() -> None:
    with pytest.raises(NiqeUnavailableError, match="at least 96"):
        score_niqe(np.zeros((32, 64, 3), dtype=np.float64))


def test_png_export_round_trip() -> None:
    image = np.zeros((48, 64, 3), dtype=np.float64)
    image[..., 0] = 1.0
    data = image_to_png_bytes(image)
    uploaded = load_uploaded_image(data)

    assert uploaded.image.shape == image.shape
    assert uploaded.image[..., 0].min() == pytest.approx(1.0)
    assert uploaded.image[..., 1].max() == pytest.approx(0.0)
