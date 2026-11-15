"""Tune classical denoisers on validation splits and freeze parameters."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lowlight.classical.tuning import (  # noqa: E402
    TUNING_COLUMNS,
    TuningSample,
    evaluate_config,
    iter_grid,
    prepare_noisy,
    select_fixed,
    select_per_level,
)
from lowlight.metrics import PsnrMetric, SsimMetric  # noqa: E402
from lowlight.utils.io import load_rgb, read_csv_rows, write_csv_rows  # noqa: E402
from lowlight.utils.noise import load_noise_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid", type=Path, default=ROOT / "configs" / "classical" / "denoising_grid.yaml"
    )
    parser.add_argument(
        "--noise-config", type=Path, default=ROOT / "configs" / "data" / "noise.yaml"
    )
    parser.add_argument("--images-per-dataset", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "results" / "classical_denoising"
    )
    return parser.parse_args()


def select_scene_rows(
    rows: list[dict[str, str]], count: int
) -> list[dict[str, str]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["scene_id"], []).append(row)
    scene_ids = sorted(groups)
    if count >= len(scene_ids):
        return [sorted(groups[scene])[0] for scene in scene_ids]
    step = len(scene_ids) / count
    selected = []
    for index in range(count):
        scene = scene_ids[min(int(index * step), len(scene_ids) - 1)]
        selected.append(sorted(groups[scene], key=lambda row: row["low_path"])[0])
    return selected


def load_samples(per_dataset: int) -> list[TuningSample]:
    samples = []
    for name in ("lol_val.csv", "lolv2_val.csv"):
        rows = select_scene_rows(
            read_csv_rows(ROOT / "data" / "splits" / name), per_dataset
        )
        for row in rows:
            samples.append(
                TuningSample(
                    dataset=row["dataset"],
                    split=row["split"],
                    image_id=row["low_path"],
                    clean=load_rgb(ROOT / row["low_path"]),
                )
            )
    return samples


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    grid_payload = yaml.safe_load(args.grid.read_text(encoding="utf-8"))
    levels = [str(level) for level in grid_payload.get("levels", [])]
    methods = args.methods or list(grid_payload["methods"])
    unsupported = [method for method in methods if method not in grid_payload["methods"]]
    if unsupported:
        raise SystemExit(f"no grid for methods: {unsupported}")

    noise_config = load_noise_config(args.noise_config)
    samples = load_samples(args.images_per_dataset)
    print(f"samples={len(samples)} levels={levels} methods={methods}")
    noisy = prepare_noisy(samples, noise_config, levels, args.seed)
    targets = [sample.clean for sample in samples]
    metrics = {"psnr": PsnrMetric(), "ssim": SsimMetric()}

    rows: list[dict[str, object]] = []
    for method in methods:
        configs = iter_grid(grid_payload["methods"][method])
        print(f"tuning {method}: {len(configs)} configs")
        for index, params in enumerate(configs, start=1):
            rows.extend(evaluate_config(method, params, noisy, metrics, targets))
            if index % 5 == 0 or index == len(configs):
                print(f"  {method} {index}/{len(configs)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tuning_csv = args.out_dir / "tuning.csv"
    write_csv_rows(tuning_csv, rows, TUNING_COLUMNS)

    selected = select_per_level(rows, methods, levels)
    frozen = {
        "frozen": True,
        "frozen_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "selected_on": ["lol_val", "lolv2_val"],
        "criterion": "psnr > ssim > inference time",
        "tuning_levels": levels,
        "noise_config": str(args.noise_config.relative_to(ROOT)),
        "noise_config_sha256": sha256(args.noise_config),
        "methods": {
            method: {
                level: {
                    "params": json.loads(str(row["params"])),
                    "mean_psnr": float(row["mean_psnr"]),
                    "std_psnr": float(row["std_psnr"]),
                    "mean_ssim": float(row["mean_ssim"]),
                    "std_ssim": float(row["std_ssim"]),
                    "median_ms": float(row["median_ms"]),
                    "n": int(row["n"]),
                }
                for level, row in per_level.items()
            }
            for method, per_level in selected.items()
        },
    }
    frozen_path = ROOT / "configs" / "classical" / "denoising_frozen.yaml"
    frozen_path.write_text(
        yaml.safe_dump(frozen, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    fixed = select_fixed(rows)
    fixed_payload = {
        "name": str(fixed["method"]),
        "params": json.loads(str(fixed["params"])),
        "selected_on": ["lol_val", "lolv2_val"],
        "criterion": "mean PSNR on low+medium+high; ties by SSIM then time",
        "validation": {
            "level": "noisy",
            "mean_psnr": float(fixed["mean_psnr"]),
            "std_psnr": float(fixed["std_psnr"]),
            "mean_ssim": float(fixed["mean_ssim"]),
            "std_ssim": float(fixed["std_ssim"]),
            "median_ms": float(fixed["median_ms"]),
            "n": int(fixed["n"]),
        },
        "noise_config": str(args.noise_config.relative_to(ROOT)),
        "noise_config_sha256": sha256(args.noise_config),
        "frozen_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    fixed_path = ROOT / "configs" / "pipelines" / "d_fixed.yaml"
    fixed_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_path.write_text(
        yaml.safe_dump(fixed_payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    selected_json = args.out_dir / "tuning_selected.json"
    selected_json.write_text(
        json.dumps(
            {"selected": frozen["methods"], "d_fixed": fixed_payload},
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"tuning={tuning_csv}")
    print(f"frozen={frozen_path}")
    print(f"d_fixed={fixed_path}")
    print(f"selected={selected_json}")
    for method, per_level in frozen["methods"].items():
        summary = ", ".join(
            f"{level}:{per_level[level]['mean_psnr']:.2f}dB" for level in levels
        )
        print(f"  {method}: {summary}")
    print(
        "D_fixed: "
        f"{fixed_payload['name']} {fixed_payload['params']} "
        f"psnr={fixed_payload['validation']['mean_psnr']:.2f}dB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
