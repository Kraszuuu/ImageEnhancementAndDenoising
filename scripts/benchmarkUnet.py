"""Benchmark compact U-Net training steps and freeze GPU defaults."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lowlight.models import build_model, count_parameters  # noqa: E402
from lowlight.utils.io import write_csv_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "results" / "environment"
    )
    parser.add_argument(
        "--defaults", type=Path, default=ROOT / "configs" / "training" / "gpu_defaults.yaml"
    )
    return parser.parse_args()


def run_step(model, optimizer, inputs, target, *, amp: bool) -> float:
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp):
        output = model(inputs)
        loss = torch.nn.functional.l1_loss(output, target)
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def timed_step(model, optimizer, inputs, target, *, amp: bool) -> tuple[float, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    loss = run_step(model, optimizer, inputs, target, amp=amp)
    torch.cuda.synchronize()
    return (time.perf_counter() - started) * 1000.0, loss


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    device = torch.device("cuda")
    torch.manual_seed(0)
    rows = []
    grid = product(
        (32, 48, 64), (192, 256), (2, 4, 8), (True, False)
    )
    for base, patch, batch, amp in grid:
        model = build_model({"base_channels": base, "depth": 4}).to(device).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        inputs = torch.rand(batch, 3, patch, patch, device=device)
        target = torch.rand_like(inputs)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.warmup):
            run_step(model, optimizer, inputs, target, amp=amp)
        samples = []
        losses = []
        for _ in range(args.repeats):
            step_ms, loss = timed_step(model, optimizer, inputs, target, amp=amp)
            samples.append(step_ms)
            losses.append(loss)
        peak = torch.cuda.max_memory_allocated() / 2**20
        rows.append(
            {
                "base_channels": base,
                "depth": 4,
                "params": count_parameters(model),
                "patch": patch,
                "batch": batch,
                "amp": amp,
                "repeat": len(samples),
                "median_ms": statistics.median(samples),
                "per_image_ms": statistics.median(samples) / batch,
                "min_ms": min(samples),
                "max_ms": max(samples),
                "peak_vram_mib": round(peak, 1),
                "oom": False,
                "loss": losses[-1],
            }
        )
        print(
            f"base={base} patch={patch} batch={batch} amp={amp} "
            f"median={rows[-1]['median_ms']:.1f} ms peak={peak:.0f} MiB"
        )
        del model, optimizer, inputs, target
        torch.cuda.empty_cache()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        args.out_dir / "unet_benchmark.csv",
        rows,
        [
            "base_channels",
            "depth",
            "params",
            "patch",
            "batch",
            "amp",
            "repeat",
            "median_ms",
            "per_image_ms",
            "min_ms",
            "max_ms",
            "peak_vram_mib",
            "oom",
            "loss",
        ],
    )

    selected = next(
        row
        for row in rows
        if row["base_channels"] == 32
        and row["patch"] == 256
        and row["batch"] == 4
        and row["amp"]
    )
    defaults = {
        "device": "cuda",
        "amp": True,
        "patch_size": 256,
        "batch_size": 4,
        "num_workers": 4,
        "grad_accum_steps": 1,
        "deterministic": True,
        "benchmark": {
            "source": "results/environment/unet_benchmark.csv",
            "base_channels": 32,
            "depth": 4,
            "repeats": args.repeats,
            "median_step_ms": round(selected["median_ms"], 2),
            "peak_vram_mib": selected["peak_vram_mib"],
            "note": "chosen for Stage 03; measured on RTX 3060 12 GB",
        },
    }
    args.defaults.parent.mkdir(parents=True, exist_ok=True)
    args.defaults.write_text(
        "# Frozen GPU defaults derived from results/environment/unet_benchmark.csv.\n"
        + yaml.safe_dump(defaults, sort_keys=False),
        encoding="utf-8",
    )
    environment = {
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": torch.cuda.get_device_name(0),
        "vram_total_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20, 1),
    }
    (args.out_dir / "gpu.json").write_text(
        json.dumps(environment, indent=2) + "\n", encoding="utf-8"
    )
    print(f"benchmark={args.out_dir / 'unet_benchmark.csv'}")
    print(f"defaults={args.defaults}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
