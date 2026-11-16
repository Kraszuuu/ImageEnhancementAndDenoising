"""Tune classical brighteners on validation splits and freeze parameters."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lowlight.classical.brightening_tuning import (  # noqa: E402
    BRIGHTENING_COLUMNS,
    BrighteningSample,
    evaluate_brightener,
    select_per_track,
)
from lowlight.classical.denoisers import build_denoiser  # noqa: E402
from lowlight.classical.tuning import iter_grid  # noqa: E402
from lowlight.metrics import (  # noqa: E402
    LpipsMetric,
    PsnrMetric,
    SsimMetric,
    resolve_device,
)
from lowlight.utils.io import load_rgb, read_csv_rows, write_csv_rows  # noqa: E402

VAL_SPLITS = ("lol_val.csv", "lolv2_val.csv")
TRACKS = ("d_fixed", "raw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid",
        type=Path,
        default=ROOT / "configs" / "classical" / "brightening_grid.yaml",
    )
    parser.add_argument(
        "--d-fixed",
        type=Path,
        default=ROOT / "configs" / "pipelines" / "d_fixed.yaml",
    )
    parser.add_argument("--images-per-dataset", type=int, default=24)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "results" / "classical_brightening"
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def load_samples(per_dataset: int) -> list[BrighteningSample]:
    samples = []
    for name in VAL_SPLITS:
        rows = select_scene_rows(
            read_csv_rows(ROOT / "data" / "splits" / name), per_dataset
        )
        for row in rows:
            samples.append(
                BrighteningSample(
                    dataset=row["dataset"],
                    split=row["split"],
                    image_id=row["low_path"],
                    target=load_rgb(ROOT / row["gt_path"]),
                )
            )
    return samples


def main() -> int:
    args = parse_args()
    grid_payload = yaml.safe_load(args.grid.read_text(encoding="utf-8"))
    tracks = [str(track) for track in grid_payload.get("tracks", TRACKS)]
    methods = args.methods or list(grid_payload["methods"])
    unsupported = [method for method in methods if method not in grid_payload["methods"]]
    if unsupported:
        raise SystemExit(f"no grid for methods: {unsupported}")
    unknown_tracks = [track for track in tracks if track not in TRACKS]
    if unknown_tracks:
        raise SystemExit(f"unknown tracks: {unknown_tracks}")

    d_fixed_payload = yaml.safe_load(args.d_fixed.read_text(encoding="utf-8"))
    denoiser = build_denoiser(
        str(d_fixed_payload["name"]), dict(d_fixed_payload["params"])
    )
    samples = load_samples(args.images_per_dataset)
    targets = [sample.target for sample in samples]
    device = resolve_device(None if args.device == "auto" else args.device)
    metrics = {
        "psnr": PsnrMetric(),
        "ssim": SsimMetric(),
        "lpips": LpipsMetric(device=device),
    }

    print(
        f"samples={len(samples)} methods={methods} tracks={tracks} device={device}"
    )
    raw_sources: list[np.ndarray] = []
    fixed_sources: list[np.ndarray] = []
    fixed_times: list[float] = []
    for index, sample in enumerate(samples, start=1):
        low = load_rgb(ROOT / sample.image_id)
        raw_sources.append(low)
        start = time.perf_counter()
        fixed_sources.append(denoiser.denoise(low))
        fixed_times.append((time.perf_counter() - start) * 1000.0)
        if index % 10 == 0 or index == len(samples):
            print(f"  D_fixed {index}/{len(samples)}")
    sources_by_track = {"d_fixed": fixed_sources, "raw": raw_sources}

    rows: list[dict[str, object]] = []
    for track in tracks:
        for method in methods:
            configs = iter_grid(grid_payload["methods"][method])
            print(f"tuning {track}/{method}: {len(configs)} configs")
            for index, params in enumerate(configs, start=1):
                rows.append(
                    evaluate_brightener(
                        track,
                        method,
                        params,
                        sources_by_track[track],
                        targets,
                        metrics,
                    )
                )
                if index % 5 == 0 or index == len(configs):
                    print(f"  {method} {index}/{len(configs)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tuning_csv = args.out_dir / "tuning.csv"
    write_csv_rows(tuning_csv, rows, BRIGHTENING_COLUMNS)

    selected = select_per_track(rows, tracks, methods)
    frozen = {
        "frozen": True,
        "frozen_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "selected_on": ["lol_val", "lolv2_val"],
        "criterion": "mean PSNR > mean SSIM > mean LPIPS > median time",
        "tuning_images_per_dataset": args.images_per_dataset,
        "n_tuning_images": len(samples),
        "grid": str(args.grid.relative_to(ROOT)),
        "grid_sha256": sha256(args.grid),
        "d_fixed": {
            "path": str(args.d_fixed.relative_to(ROOT)),
            "sha256": sha256(args.d_fixed),
            "name": d_fixed_payload["name"],
            "params": d_fixed_payload["params"],
            "median_ms": float(np.median(fixed_times)),
        },
        "tracks": {
            track: {
                "label": "D_fixed(L)" if track == "d_fixed" else "L",
                "methods": {
                    method: {
                        "params": json.loads(str(selected[track][method]["params"])),
                        "validation": {
                            "mean_psnr": float(selected[track][method]["mean_psnr"]),
                            "std_psnr": float(selected[track][method]["std_psnr"]),
                            "mean_ssim": float(selected[track][method]["mean_ssim"]),
                            "std_ssim": float(selected[track][method]["std_ssim"]),
                            "mean_lpips": float(
                                selected[track][method]["mean_lpips"]
                            ),
                            "std_lpips": float(selected[track][method]["std_lpips"]),
                            "median_ms": float(selected[track][method]["median_ms"]),
                            "n": int(selected[track][method]["n"]),
                        },
                    }
                    for method in methods
                },
            }
            for track in tracks
        },
    }
    for track in tracks:
        best = max(
            methods,
            key=lambda method: float(
                frozen["tracks"][track]["methods"][method]["validation"]["mean_psnr"]
            ),
        )
        frozen["tracks"][track]["best_method"] = best

    frozen_path = ROOT / "configs" / "classical" / "brightening_frozen.yaml"
    frozen_path.write_text(
        yaml.safe_dump(frozen, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    selected_json = args.out_dir / "tuning_selected.json"
    selected_json.write_text(
        json.dumps(
            {
                "selected": selected,
                "d_fixed_timing": {
                    "mean_ms": float(np.mean(fixed_times)),
                    "median_ms": float(np.median(fixed_times)),
                    "n": len(fixed_times),
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"tuning={tuning_csv}")
    print(f"frozen={frozen_path}")
    print(f"selected={selected_json}")
    for track in tracks:
        for method in methods:
            row = selected[track][method]
            print(
                f"  {track}/{method}: psnr={float(row['mean_psnr']):.2f}dB "
                f"ssim={float(row['mean_ssim']):.4f} "
                f"lpips={float(row['mean_lpips']):.4f} "
                f"median={float(row['median_ms']):.1f}ms "
                f"params={row['params']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
