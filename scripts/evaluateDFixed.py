"""Evaluate the frozen Stage 02 D_fixed BM3D reference on test (evaluation only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lowlight.classical.denoisers import build_denoiser  # noqa: E402
from lowlight.metrics import (  # noqa: E402
    LpipsMetric,
    NiqeMetric,
    PsnrMetric,
    SsimMetric,
    resolve_device,
    summarize,
)
from lowlight.metrics.runner import PER_IMAGE_COLUMNS, SUMMARY_COLUMNS  # noqa: E402
from lowlight.utils.io import load_rgb, read_csv_rows, write_csv_rows  # noqa: E402
from lowlight.utils.noise import load_noise_config  # noqa: E402

TEST_SPLITS = ("lol_test.csv", "lolv2_test.csv")
LEVELS = ("clean", "low", "medium", "high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--d-fixed", type=Path, default=ROOT / "configs" / "pipelines" / "d_fixed.yaml"
    )
    parser.add_argument(
        "--noise-config", type=Path, default=ROOT / "configs" / "data" / "noise.yaml"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "denoising_cnn" / "evaluation_d_fixed",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def format_cell(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.6f}"
    return value


def batch_lpips(
    metric: LpipsMetric, predictions: list[np.ndarray], targets: list[np.ndarray], batch: int = 8
) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(predictions), batch):
        pred = np.stack(
            [np.transpose(item, (2, 0, 1)) for item in predictions[start : start + batch]]
        ).astype(np.float32)
        ref = np.stack(
            [np.transpose(item, (2, 0, 1)) for item in targets[start : start + batch]]
        ).astype(np.float32)
        pred_t = torch.from_numpy(pred).to(metric.device) * 2.0 - 1.0
        ref_t = torch.from_numpy(ref).to(metric.device) * 2.0 - 1.0
        with (
            torch.inference_mode(),
            torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True),
        ):
            values = metric.model(pred_t, ref_t).reshape(-1).cpu().tolist()
        scores.extend(float(value) for value in values)
    return scores


def batch_niqe(
    metric: NiqeMetric, predictions: list[np.ndarray], batch: int = 8
) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(predictions), batch):
        array = np.stack(
            [np.transpose(item, (2, 0, 1)) for item in predictions[start : start + batch]]
        ).astype(np.float32)
        tensor = torch.from_numpy(array).to(metric.device)
        with torch.inference_mode():
            values = metric.model(tensor).reshape(-1).cpu().tolist()
        scores.extend(float(value) for value in values)
    return scores


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    per_image_path = out_dir / "per_image.csv"
    if per_image_path.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {per_image_path} without --force")
    config = yaml.safe_load(args.d_fixed.read_text(encoding="utf-8"))
    noise_config = load_noise_config(args.noise_config)
    rows = []
    for name in TEST_SPLITS:
        rows.extend(read_csv_rows(ROOT / "data" / "splits" / name))
    device = resolve_device(None if args.device == "auto" else args.device)
    metrics = {"psnr": PsnrMetric(), "ssim": SsimMetric()}
    lpips_metric = LpipsMetric(device=device)
    niqe_metric = NiqeMetric(device=device)
    all_metrics = [*metrics.values(), lpips_metric, niqe_metric]
    denoiser = build_denoiser(config["name"], config["params"])
    result_rows: list[dict[str, object]] = []
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    print(f"images={len(rows)} method={config['name']} sigma={config['params']['sigma_psd']}")
    for index, row in enumerate(rows, start=1):
        clean = load_rgb(ROOT / row["low_path"])
        height, width = clean.shape[:2]
        for level in LEVELS:
            noisy = noise_config.add(clean, level, image_id=row["low_path"])
            started = time.perf_counter()
            prediction = denoiser.denoise(noisy)
            elapsed = (time.perf_counter() - started) * 1000.0
            scores = {
                name: float(metric.compute(prediction, clean))
                for name, metric in metrics.items()
            }
            result_rows.append(
                {
                    "run_id": "stage03-d-fixed-reference",
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "method": "d_fixed",
                    "seed": 0,
                    "noise_level": level,
                    "image_id": row["low_path"],
                    "psnr": scores["psnr"],
                    "ssim": scores["ssim"],
                    "lpips": 0.0,
                    "niqe": 0.0,
                    "inference_ms": elapsed,
                    "width": width,
                    "height": height,
                }
            )
            predictions.append(prediction)
            targets.append(clean)
        if index % 10 == 0 or index == len(rows):
            print(f"  {index}/{len(rows)}")
    scores_lpips = batch_lpips(lpips_metric, predictions, targets)
    scores_niqe = batch_niqe(niqe_metric, predictions)
    for row, lpips_score, niqe_score in zip(
        result_rows, scores_lpips, scores_niqe, strict=True
    ):
        row["lpips"] = lpips_score
        row["niqe"] = niqe_score

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        per_image_path,
        [{key: format_cell(value) for key, value in row.items()} for row in result_rows],
        PER_IMAGE_COLUMNS,
    )
    summary_rows = summarize(result_rows)
    write_csv_rows(
        out_dir / "summary.csv",
        [{key: format_cell(value) for key, value in row.items()} for row in summary_rows],
        SUMMARY_COLUMNS,
    )
    timing_rows = []
    for dataset in sorted({str(row["dataset"]) for row in result_rows}):
        for level in LEVELS:
            values = np.asarray(
                [
                    float(row["inference_ms"])
                    for row in result_rows
                    if row["dataset"] == dataset and row["noise_level"] == level
                ]
            )
            timing_rows.append(
                {
                    "method": "d_fixed",
                    "dataset": dataset,
                    "device": "cpu",
                    "noise_level": level,
                    "mean_ms": float(values.mean()),
                    "std_ms": float(values.std(ddof=1)),
                    "median_ms": float(np.median(values)),
                    "n": int(values.size),
                }
            )
    write_csv_rows(
        out_dir / "timing.csv",
        [{key: format_cell(value) for key, value in row.items()} for row in timing_rows],
        ["method", "dataset", "device", "noise_level", "mean_ms", "std_ms", "median_ms", "n"],
    )
    split_manifest = json.loads(
        (ROOT / "data" / "splits" / "split_manifest.json").read_text(encoding="utf-8")
    )
    metadata = {
        "run_id": "stage03-d-fixed-reference",
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "python": platform.python_version(),
        "test_used": True,
        "purpose": "frozen D_fixed test reference; no tuning",
        "splits": list(TEST_SPLITS),
        "n_images": len(rows),
        "n_rows": len(result_rows),
        "d_fixed": config,
        "d_fixed_sha256": sha256(args.d_fixed),
        "noise_config_sha256": sha256(args.noise_config),
        "split_hashes": split_manifest.get("files_sha256", {}),
        "metrics": [metric.metadata() for metric in all_metrics],
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "test_lock.json").write_text(
        json.dumps(
            {
                "used_at_utc": metadata["created_at_utc"],
                "split_hashes": metadata["split_hashes"],
                "per_image_sha256": sha256(per_image_path),
                "summary_sha256": sha256(out_dir / "summary.csv"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"per_image={per_image_path}")
    print(f"summary={out_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
