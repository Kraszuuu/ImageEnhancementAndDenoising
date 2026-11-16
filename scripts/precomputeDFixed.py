"""Precompute frozen D_fixed outputs for CNN brightening training."""

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

from lowlight.classical.denoisers import build_denoiser  # noqa: E402
from lowlight.training.brightening_datasets import (  # noqa: E402
    BrighteningRow,
    d_fixed_cache_path,
    load_brightening_rows,
    select_evenly,
)
from lowlight.utils.io import load_rgb  # noqa: E402

DEFAULT_SPLITS = (
    "lol_train.csv",
    "lolv2_train.csv",
    "lol_val.csv",
    "lolv2_val.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="*", default=list(DEFAULT_SPLITS))
    parser.add_argument(
        "--d-fixed",
        type=Path,
        default=ROOT / "configs" / "pipelines" / "d_fixed.yaml",
    )
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "outputs" / "d_fixed_cache")
    parser.add_argument("--limit-per-dataset", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    args = parse_args()
    d_fixed = yaml.safe_load(args.d_fixed.read_text(encoding="utf-8"))
    denoiser = build_denoiser(str(d_fixed["name"]), dict(d_fixed["params"]))
    split_paths = [ROOT / "data" / "splits" / name for name in args.splits]
    rows: list[BrighteningRow] = load_brightening_rows(ROOT, split_paths)
    if args.limit_per_dataset > 0:
        rows = select_evenly(rows, args.limit_per_dataset)
    unique: dict[str, BrighteningRow] = {}
    for row in rows:
        unique.setdefault(row.image_id, row)
    selected = sorted(unique.values(), key=lambda row: row.image_id)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"images={len(selected)} cache={args.cache_dir} "
        f"d_fixed={d_fixed['name']} {d_fixed['params']}"
    )
    started = time.perf_counter()
    entries = []
    computed = 0
    skipped = 0
    for index, row in enumerate(selected, start=1):
        path = d_fixed_cache_path(args.cache_dir, row.image_id)
        if path.exists() and not args.force:
            skipped += 1
        else:
            image = load_rgb(row.low_path)
            fixed = denoiser.denoise(image)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, fixed.astype(np.float16))
            computed += 1
        entries.append(
            {
                "image_id": row.image_id,
                "dataset": row.dataset,
                "split": row.split,
                "cache_file": path.name,
                "sha256": sha256(path),
            }
        )
        if computed and computed % 10 == 0:
            elapsed = time.perf_counter() - started
            print(
                f"  {index}/{len(selected)} computed={computed} skipped={skipped} "
                f"elapsed={elapsed:.0f}s"
            )

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "d_fixed_config": str(args.d_fixed.relative_to(ROOT)),
        "d_fixed_config_sha256": sha256(args.d_fixed),
        "d_fixed_name": d_fixed["name"],
        "d_fixed_params": d_fixed["params"],
        "dtype": "float16",
        "n_images": len(entries),
        "computed": computed,
        "skipped": skipped,
        "entries": entries,
    }
    (args.cache_dir / "cache_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"manifest={args.cache_dir / 'cache_manifest.json'} "
        f"computed={computed} skipped={skipped} "
        f"seconds={time.perf_counter() - started:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
