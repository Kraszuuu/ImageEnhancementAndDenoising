"""Evaluate frozen Stage 03 CNN checkpoints on the locked test splits (run once)."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lowlight.metrics import (  # noqa: E402
    LpipsMetric,
    NiqeMetric,
    PsnrMetric,
    SsimMetric,
    resolve_device,
    summarize,
)
from lowlight.metrics.runner import PER_IMAGE_COLUMNS, SUMMARY_COLUMNS  # noqa: E402
from lowlight.models import build_model, count_parameters, denoise_full_image  # noqa: E402
from lowlight.training.checkpointing import CheckpointManager  # noqa: E402
from lowlight.utils.io import load_rgb, read_csv_rows, write_csv_rows  # noqa: E402
from lowlight.utils.noise import load_noise_config  # noqa: E402

TEST_SPLITS = ("lol_test.csv", "lolv2_test.csv")
LEVELS = ("clean", "low", "medium", "high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=Path,
        default=[
            ROOT / "checkpoints" / "denoising_cnn" / "seed0" / "best.pt",
            ROOT / "checkpoints" / "denoising_cnn" / "seed1" / "best.pt",
            ROOT / "checkpoints" / "denoising_cnn" / "seed2" / "best.pt",
        ],
    )
    parser.add_argument(
        "--noise-config", type=Path, default=ROOT / "configs" / "data" / "noise.yaml"
    )
    parser.add_argument("--splits", nargs="*", default=list(TEST_SPLITS))
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "results" / "denoising_cnn" / "evaluation"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timing-iters", type=int, default=50)
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


def load_rows(splits: list[str]) -> list[dict[str, str]]:
    rows = []
    for name in splits:
        rows.extend(read_csv_rows(ROOT / "data" / "splits" / name))
    return rows


def metric_values(
    metrics: dict[str, object], prediction: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    return {
        name: float(metric.compute(prediction, target))  # type: ignore[attr-defined]
        for name, metric in metrics.items()
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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
    if tuple(args.splits) != TEST_SPLITS:
        raise SystemExit("Stage 03 evaluation is restricted to the locked test splits")
    if per_image_path.exists() and not args.force:
        raise SystemExit(
            f"refusing to overwrite {per_image_path}; test is locked "
            "(use --force only with a documented reason)"
        )
    for checkpoint in args.checkpoints:
        if not checkpoint.exists():
            raise SystemExit(f"missing checkpoint: {checkpoint}")

    noise_config = load_noise_config(args.noise_config)
    rows = load_rows(args.splits)
    device = resolve_device(None if args.device == "auto" else args.device)
    metrics = {"psnr": PsnrMetric(), "ssim": SsimMetric()}
    lpips_metric = LpipsMetric(device=device)
    niqe_metric = NiqeMetric(device=device)
    all_metrics = [*metrics.values(), lpips_metric, niqe_metric]
    test_images = {
        row["low_path"]: load_rgb(ROOT / row["low_path"]) for row in rows
    }
    test_noisy = {
        (row["low_path"], level): noise_config.add(
            test_images[row["low_path"]], level, image_id=row["low_path"]
        )
        for row in rows
        for level in LEVELS
    }

    print(f"images={len(rows)} seeds={len(args.checkpoints)} levels={LEVELS} device={device}")
    result_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    checkpoint_meta = []
    for checkpoint_path in args.checkpoints:
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        seed = int(payload.get("seed", 0))
        model_config = payload["config"]["model"]
        model = build_model(model_config).to(device).eval()
        CheckpointManager.load(checkpoint_path, model, device=device)
        model.eval()
        checkpoint_meta.append(
            {
                "path": str(checkpoint_path.relative_to(ROOT)),
                "sha256": sha256(checkpoint_path),
                "seed": seed,
                "epoch": payload.get("epoch"),
                "val_psnr": payload.get("score"),
                "parameters": count_parameters(model),
            }
        )
        print(
            f"seed={seed} epoch={payload.get('epoch')} "
            f"val_psnr={float(payload.get('score', math.nan)):.3f} "
            f"params={count_parameters(model)}"
        )
        seed_rows: list[dict[str, object]] = []
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        tasks = [
            (row, level) for row in rows for level in LEVELS
        ]
        for start in range(0, len(tasks), 4):
            batch = tasks[start : start + 4]
            noisy = np.stack(
                [test_noisy[(row["low_path"], level)] for row, level in batch]
            ).astype(np.float32)
            tensor = torch.from_numpy(noisy).permute(0, 3, 1, 2).to(device)
            synchronize(device)
            started = time.perf_counter()
            with torch.no_grad():
                output = denoise_full_image(model, tensor)
            synchronize(device)
            elapsed = (time.perf_counter() - started) * 1000.0 / len(batch)
            images = output.permute(0, 2, 3, 1).float().clamp(0.0, 1.0).cpu().numpy()
            for (row, level), prediction in zip(batch, images, strict=True):
                clean = test_images[row["low_path"]]
                scores = metric_values(metrics, prediction, clean)
                height, width = clean.shape[:2]
                seed_rows.append(
                    {
                        "run_id": "stage03-cnn-denoising",
                        "dataset": row["dataset"],
                        "split": row["split"],
                        "method": "cnn_unet",
                        "seed": seed,
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
            if start % 80 == 0:
                print(f"  seed {seed}: predictions {min(start + 4, len(tasks))}/{len(tasks)}")
        lpips_scores = batch_lpips(lpips_metric, predictions, targets)
        niqe_scores = batch_niqe(niqe_metric, predictions)
        for row, lpips_score, niqe_score in zip(
            seed_rows, lpips_scores, niqe_scores, strict=True
        ):
            row["lpips"] = lpips_score
            row["niqe"] = niqe_score
        result_rows.extend(seed_rows)
        print(f"  seed {seed}: metrics done")

        for dataset in sorted({row["dataset"] for row in rows}):
            dataset_row = next(row for row in rows if row["dataset"] == dataset)
            for level in LEVELS:
                noisy = noise_config.add(
                    test_images[dataset_row["low_path"]],
                    level,
                    image_id=dataset_row["low_path"],
                )
                tensor = torch.from_numpy(noisy).permute(2, 0, 1).unsqueeze(0).float().to(device)
                with torch.no_grad():
                    for _ in range(args.warmup):
                        denoise_full_image(model, tensor)
                    synchronize(device)
                    samples = []
                    for _ in range(args.timing_iters):
                        synchronize(device)
                        started = time.perf_counter()
                        denoise_full_image(model, tensor)
                        synchronize(device)
                        samples.append((time.perf_counter() - started) * 1000.0)
                array = np.asarray(samples)
                timing_rows.append(
                    {
                        "method": "cnn_unet",
                        "dataset": dataset,
                        "device": device.type,
                        "seed": seed,
                        "noise_level": level,
                        "mean_ms": float(array.mean()),
                        "std_ms": float(array.std(ddof=1)),
                        "median_ms": float(np.median(array)),
                        "n": int(array.size),
                        "protocol": f"batch=1 warmup={args.warmup} iters={args.timing_iters}",
                    }
                )
                del tensor

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
    write_csv_rows(
        out_dir / "timing.csv",
        [{key: format_cell(value) for key, value in row.items()} for row in timing_rows],
        [
            "method",
            "dataset",
            "device",
            "seed",
            "noise_level",
            "mean_ms",
            "std_ms",
            "median_ms",
            "n",
            "protocol",
        ],
    )

    split_manifest = json.loads(
        (ROOT / "data" / "splits" / "split_manifest.json").read_text(encoding="utf-8")
    )
    training_config = ROOT / "configs" / "training" / "denoising.yaml"
    model_config = ROOT / "configs" / "models" / "unet_denoising.yaml"
    metadata = {
        "run_id": "stage03-cnn-denoising",
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "python": platform.python_version(),
        "device": str(device),
        "torch": torch.__version__,
        "test_used": True,
        "splits": list(args.splits),
        "n_images": len(rows),
        "n_rows": len(result_rows),
        "noise_config": str(args.noise_config.relative_to(ROOT)),
        "noise_config_sha256": sha256(args.noise_config),
        "training_config": str(training_config.relative_to(ROOT)),
        "training_config_sha256": sha256(training_config),
        "model_config": str(model_config.relative_to(ROOT)),
        "model_config_sha256": sha256(model_config),
        "checkpoints": checkpoint_meta,
        "split_hashes": split_manifest.get("files_sha256", {}),
        "metrics": [metric.metadata() for metric in all_metrics],
        "aggregation": "bootstrap 1000, ci 0.95, seed 0; non-finite excluded",
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
    print(f"timing={out_dir / 'timing.csv'}")
    print(f"metadata={out_dir / 'run_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
