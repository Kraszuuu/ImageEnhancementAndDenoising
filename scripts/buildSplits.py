"""Build frozen train/validation/test splits from the combined manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lowlight.data.manifest import read_csv  # noqa: E402
from lowlight.data.splits import build_split_outputs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "manifests" / "combined.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.18)
    parser.add_argument("--min-val-pairs", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_csv(args.manifest)
    manifest = build_split_outputs(
        rows,
        args.out_dir,
        seed=args.seed,
        val_fraction=args.val_fraction,
        min_val_pairs=args.min_val_pairs,
    )
    for dataset, counts in manifest["counts"].items():
        print(
            f"{dataset}: train={counts['train']} val={counts['val']} "
            f"test={counts['test']}"
        )
        overlap = manifest["scene_overlap"][dataset]
        print(
            f"  scenes: train_val={overlap['train_val_scenes']} "
            f"test={overlap['test_scenes']} shared={overlap['shared_scenes']}"
        )
    print(f"split manifest -> {args.out_dir / 'split_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
