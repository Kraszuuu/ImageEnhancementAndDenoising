from __future__ import annotations

import pytest

from lowlight.evaluation.aggregation import aggregate_rows, filter_rows, per_image_mean


def _row(image_id: str, seed: str, psnr: str) -> dict[str, str]:
    return {
        "source": "synthetic",
        "task": "denoising",
        "variant": "cnn",
        "method": "m",
        "track": "",
        "dataset": "d",
        "split": "test",
        "noise_level": "high",
        "seed": seed,
        "image_id": image_id,
        "psnr": psnr,
        "ssim": "0.5",
        "lpips": "0.2",
        "niqe": "4.0",
        "inference_ms": "1.0",
        "brightness_error": "",
        "color_cast": "",
        "clip_low_fraction": "",
        "clip_high_fraction": "",
    }


def test_per_image_mean_averages_seeds() -> None:
    rows = [_row("a", "0", "10"), _row("a", "1", "14"), _row("b", "0", "20")]
    assert per_image_mean(rows, "psnr") == {("d", "a"): 12.0, ("d", "b"): 20.0}


def test_aggregate_uses_image_then_seed_means() -> None:
    rows = [
        _row("a", "0", "10"),
        _row("a", "1", "14"),
        _row("b", "0", "20"),
        _row("b", "1", "24"),
    ]
    result = aggregate_rows(rows, ["method"], metrics=["psnr"], resamples=50)
    assert len(result) == 1
    row = result[0]
    assert row["n"] == 2
    assert row["n_seeds"] == 2
    assert row["mean"] == pytest.approx(17.0)
    assert row["seed_mean"] == pytest.approx(17.0)
    assert row["seed_std"] == pytest.approx(2.828427, rel=1e-5)


def test_filter_rows_matches_all_fields() -> None:
    rows = [_row("a", "0", "10"), {**_row("b", "0", "10"), "dataset": "other"}]
    assert [row["image_id"] for row in filter_rows(rows, dataset="d")] == ["a"]
    assert filter_rows(rows, method="missing") == []
