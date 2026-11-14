from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from lowlight.metrics.base import BaseMetric
from lowlight.metrics.runner import (
    PER_IMAGE_COLUMNS,
    SUMMARY_COLUMNS,
    ImagePair,
    MetricsRunner,
    summarize,
)


class EchoPsnr(BaseMetric):
    name = "psnr"

    def compute(self, prediction: object, target: object | None) -> float:
        del target
        return float(prediction)


class DoubleSsim(BaseMetric):
    name = "ssim"

    def compute(self, prediction: object, target: object | None) -> float:
        del target
        return 2.0 * float(prediction)


def _pairs(values: list[float]) -> list[ImagePair]:
    target = np.zeros((4, 5, 3))
    return [
        ImagePair(image_id=f"img{index}", prediction=value, target=target, inference_ms=1.5)
        for index, value in enumerate(values)
    ]


def _runner(seed: int = 0) -> MetricsRunner:
    return MetricsRunner(
        [EchoPsnr(), DoubleSsim()],
        run_id="run-test",
        method="identity",
        dataset="lolv2-real",
        split="val",
        noise_level="clean",
        seed=seed,
        split_hash="abc123",
        bootstrap_resamples=100,
        bootstrap_seed=0,
    )


def test_write_formats(tmp_path: Path) -> None:
    runner = _runner()
    rows = runner.evaluate(_pairs([1.0, 2.0, 3.0, 4.0]))
    per_image, summary, metadata = runner.write(rows, tmp_path)

    with per_image.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == PER_IMAGE_COLUMNS
        records = list(reader)
    assert len(records) == 4
    assert records[0]["image_id"] == "img0"
    assert records[0]["psnr"] == "1.000000"
    assert records[0]["ssim"] == "2.000000"
    assert records[0]["lpips"] == ""
    assert records[0]["niqe"] == ""
    assert records[0]["width"] == "5"
    assert records[0]["height"] == "4"

    with summary.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == SUMMARY_COLUMNS
        summary_rows = list(reader)
    psnr_row = next(row for row in summary_rows if row["metric"] == "psnr")
    assert psnr_row["mean"] == "2.500000"
    assert psnr_row["median"] == "2.500000"
    assert psnr_row["n"] == "4"
    assert 1.0 <= float(psnr_row["ci95_low"]) <= 2.5
    assert 2.5 <= float(psnr_row["ci95_high"]) <= 4.0

    payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert payload["run_id"] == "run-test"
    assert payload["split_hash"] == "abc123"
    assert [item["name"] for item in payload["metrics"]] == ["psnr", "ssim"]
    assert payload["n_images"] == 4


def test_summary_pools_seeds() -> None:
    rows = []
    for seed, values in {0: [1.0, 2.0], 1: [3.0, 4.0]}.items():
        for index, value in enumerate(values):
            rows.append(
                {
                    "method": "m",
                    "dataset": "d",
                    "split": "val",
                    "noise_level": "clean",
                    "seed": seed,
                    "image_id": f"{seed}-{index}",
                    "psnr": value,
                    "ssim": "",
                    "lpips": "",
                    "niqe": "",
                }
            )
    summary = summarize(rows, resamples=100, seed=0)
    pooled = next(
        row
        for row in summary
        if row["metric"] == "psnr" and row["seed"] == "all"
    )
    assert pooled["n"] == 4
    assert pooled["mean"] == pytest.approx(2.5)
    assert summarize(rows, resamples=100, seed=0) == summary


def test_non_finite_values_are_excluded_from_aggregates() -> None:
    rows = [
        {
            "method": "m",
            "dataset": "d",
            "split": "val",
            "noise_level": "clean",
            "seed": 0,
            "psnr": float("inf"),
            "ssim": "",
            "lpips": "",
            "niqe": "",
        }
    ]
    summary = summarize(rows, resamples=10)
    psnr_row = next(row for row in summary if row["metric"] == "psnr")
    assert psnr_row["n"] == 0
    assert psnr_row["mean"] == ""


def test_duplicate_metric_is_rejected() -> None:
    with pytest.raises(ValueError):
        MetricsRunner(
            [EchoPsnr(), EchoPsnr()],
            run_id="r",
            method="m",
            dataset="d",
            split="val",
            noise_level="clean",
        )
