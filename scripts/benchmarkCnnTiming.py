"""Measure frozen CNN inference time after GPU warm-up."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lowlight.models import build_model, denoise_full_image  # noqa: E402
from lowlight.training.checkpointing import CheckpointManager  # noqa: E402
from lowlight.utils.io import load_rgb, read_csv_rows, write_csv_rows  # noqa: E402
from lowlight.utils.noise import load_noise_config  # noqa: E402

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
        "--out", type=Path, default=ROOT / "results" / "denoising_cnn" / "evaluation" / "timing.csv"
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=50)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    device = torch.device("cuda")
    noise_config = load_noise_config(ROOT / "configs" / "data" / "noise.yaml")
    rows = read_csv_rows(ROOT / "data" / "splits" / "lol_test.csv") + read_csv_rows(
        ROOT / "data" / "splits" / "lolv2_test.csv"
    )
    dataset_rows = {
        dataset: next(row for row in rows if row["dataset"] == dataset)
        for dataset in sorted({row["dataset"] for row in rows})
    }
    timing_rows = []
    for checkpoint in args.checkpoints:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        seed = int(payload.get("seed", 0))
        model = build_model(payload["config"]["model"]).to(device).eval()
        CheckpointManager.load(checkpoint, model, device=device)
        model.eval()
        for dataset, row in dataset_rows.items():
            clean = load_rgb(ROOT / row["low_path"])
            for level in LEVELS:
                noisy = noise_config.add(clean, level, image_id=row["low_path"])
                tensor = (
                    torch.from_numpy(noisy)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    .to(device)
                )
                with torch.no_grad():
                    for _ in range(args.warmup):
                        denoise_full_image(model, tensor)
                    torch.cuda.synchronize()
                    samples = []
                    for _ in range(args.iters):
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        denoise_full_image(model, tensor)
                        torch.cuda.synchronize()
                        samples.append((time.perf_counter() - started) * 1000.0)
                array = np.asarray(samples)
                timing_rows.append(
                    {
                        "method": "cnn_unet",
                        "dataset": dataset,
                        "device": "cuda",
                        "seed": seed,
                        "noise_level": level,
                        "mean_ms": float(array.mean()),
                        "std_ms": float(array.std(ddof=1)),
                        "median_ms": float(np.median(array)),
                        "n": int(array.size),
                        "protocol": f"batch=1 warmup={args.warmup} iters={args.iters}",
                    }
                )
                print(
                    f"seed={seed} dataset={dataset} level={level} "
                    f"median={np.median(array):.2f} ms"
                )
                del tensor
        del model
        torch.cuda.empty_cache()
    write_csv_rows(
        args.out,
        timing_rows,
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
    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "purpose": "corrected CNN timing after global warm-up; predictions unchanged",
        "warmup": args.warmup,
        "iters": args.iters,
        "gpu": torch.cuda.get_device_name(0),
    }
    args.out.with_name("timing_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"timing={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
