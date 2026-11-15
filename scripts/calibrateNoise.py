"""Calibrate Gaussian-Poisson noise levels on the validation splits."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lowlight.utils.io import load_rgb, read_csv_rows  # noqa: E402
from lowlight.utils.noise import add_noise, stable_seed  # noqa: E402

ALPHAS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)
SIGMAS = (0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08)
TARGETS = {"low": 32.0, "medium": 26.0, "high": 20.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-per-dataset", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results" / "classical_denoising",
    )
    return parser.parse_args()


def select_rows(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["scene_id"], []).append(row)
    scene_ids = sorted(groups)
    if count >= len(scene_ids):
        return [sorted(groups[scene], key=lambda row: row["low_path"])[0] for scene in scene_ids]
    step = len(scene_ids) / count
    return [
        sorted(
            groups[scene_ids[min(int(index * step), len(scene_ids) - 1)]],
            key=lambda row: row["low_path"],
        )[0]
        for index in range(count)
    ]


def psnr(reference: np.ndarray, prediction: np.ndarray) -> float:
    mse = float(np.mean((reference - prediction) ** 2))
    return math.inf if mse == 0.0 else 10.0 * math.log10(1.0 / mse)


def main() -> int:
    args = parse_args()
    selected = []
    for split in ("lol_val.csv", "lolv2_val.csv"):
        selected.extend(
            select_rows(
                read_csv_rows(ROOT / "data" / "splits" / split),
                args.images_per_dataset,
            )
        )

    images = [
        (row["low_path"], load_rgb(ROOT / row["low_path"])) for row in selected
    ]
    print(f"calibration images={len(images)}")

    results = []
    for level in TARGETS:
        for alpha in ALPHAS:
            for sigma in SIGMAS:
                scores = []
                for image_id, clean in images:
                    noisy = add_noise(
                        clean,
                        alpha=alpha,
                        sigma=sigma,
                        seed=stable_seed(args.seed, image_id, level),
                    )
                    scores.append(psnr(clean, noisy))
                results.append(
                    {
                        "level": level,
                        "alpha": alpha,
                        "sigma": sigma,
                        "mean_psnr": float(np.mean(scores)),
                        "std_psnr": float(np.std(scores, ddof=1)),
                        "n": len(scores),
                    }
                )
        print(f"level={level}: {len(ALPHAS) * len(SIGMAS)} candidates done")

    selected_params = {}
    for level, target in TARGETS.items():
        candidates = [row for row in results if row["level"] == level]
        best = min(candidates, key=lambda row: abs(row["mean_psnr"] - target))
        selected_params[level] = best
        print(
            f"selected {level}: alpha={best['alpha']} sigma={best['sigma']} "
            f"psnr={best['mean_psnr']:.3f} (target {target})"
        )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "noise_calibration.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": args.seed,
        "images_per_dataset": args.images_per_dataset,
        "targets": TARGETS,
        "selected": selected_params,
    }
    (out_dir / "noise_calibration.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"report={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
