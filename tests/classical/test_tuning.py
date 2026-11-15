from __future__ import annotations

import numpy as np
import pytest

from lowlight.classical.tuning import (
    TuningSample,
    best_row,
    evaluate_config,
    iter_grid,
    select_fixed,
    select_per_level,
)
from lowlight.metrics import PsnrMetric, SsimMetric


def _rows(method: str = "bm3d"):
    return [
        {
            "method": method,
            "level": level,
            "params": '{"x":1}',
            "n": 10,
            "mean_psnr": psnr,
            "std_psnr": 0.1,
            "mean_ssim": ssim,
            "std_ssim": 0.01,
            "mean_ms": 1.0,
            "median_ms": 1.0,
        }
        for level, psnr, ssim in (
            ("clean", 40.0, 0.99),
            ("low", 34.0, 0.95),
            ("medium", 28.0, 0.90),
            ("high", 22.0, 0.80),
            ("all", 31.0, 0.91),
            ("noisy", 28.0, 0.88),
        )
    ]


def test_iter_grid_is_deterministic() -> None:
    grid = {"b": [1, 2], "a": [3, 4]}
    configs = iter_grid(grid)
    assert configs == [
        {"a": 3, "b": 1},
        {"a": 3, "b": 2},
        {"a": 4, "b": 1},
        {"a": 4, "b": 2},
    ]
    assert iter_grid(grid) == configs


def test_best_row_and_selection() -> None:
    rows = _rows()
    assert best_row(rows, "high")["mean_psnr"] == 22.0
    selected = select_per_level(rows, ["bm3d"], ["clean", "low"])
    assert selected["bm3d"]["clean"]["mean_psnr"] == 40.0


def test_select_fixed_ignores_clean() -> None:
    fixed = select_fixed(_rows("bm3d") + _rows("nlm"))
    assert fixed["level"] == "noisy"


def test_evaluate_config_shapes() -> None:
    samples = [
        TuningSample(
            dataset="d",
            split="val",
            image_id=f"img{index}.png",
            clean=np.random.default_rng(index).random((32, 32, 3)),
        )
        for index in range(2)
    ]
    from pathlib import Path

    from lowlight.utils.noise import load_noise_config

    root = Path(__file__).resolve().parents[2]
    config = load_noise_config(root / "configs" / "data" / "noise.yaml")
    noisy = {
        level: [
            config.add(sample.clean, level, image_id=sample.image_id)
            for sample in samples
        ]
        for level in ("clean", "low")
    }
    rows = evaluate_config(
        "gaussian",
        {"sigma": 0.8},
        noisy,
        {"psnr": PsnrMetric(), "ssim": SsimMetric()},
        [sample.clean for sample in samples],
    )
    levels = {row["level"] for row in rows}
    assert levels == {"clean", "low", "all", "noisy"}
    for row in rows:
        expected = 4 if row["level"] == "all" else 2
        assert row["n"] == expected
        assert np.isfinite(row["mean_psnr"])


def test_invalid_level() -> None:
    with pytest.raises(ValueError):
        best_row([], "clean")
