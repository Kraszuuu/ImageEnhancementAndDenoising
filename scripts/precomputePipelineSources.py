"""Precompute frozen CNN denoiser outputs for Stage 06 brightener training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lowlight.metrics import resolve_device  # noqa: E402
from lowlight.models import build_model, denoise_full_image  # noqa: E402
from lowlight.training.brightening_datasets import (  # noqa: E402
    BrighteningRow,
    load_brightening_rows,
    select_evenly,
)
from lowlight.training.checkpointing import CheckpointManager  # noqa: E402
from lowlight.training.pipeline_datasets import source_cache_path  # noqa: E402
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
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "denoising_cnn" / "seed0" / "best.pt",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "outputs" / "pipeline_source_cache" / "cnn" / "seed0",
    )
    parser.add_argument("--limit-per-dataset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def predict_batch(
    model: torch.nn.Module,
    sources: list[np.ndarray],
    device: torch.device,
) -> list[np.ndarray]:
    tensor = torch.from_numpy(np.stack(sources).astype(np.float32)).permute(0, 3, 1, 2).to(device)
    output = denoise_full_image(model, tensor)
    images = output.permute(0, 2, 3, 1).float().clamp(0.0, 1.0).cpu().numpy()
    return [image.astype(np.float64) for image in images]


def main() -> int:
    args = parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"missing checkpoint: {args.checkpoint}")
    device = resolve_device(None if args.device == "auto" else args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(dict(payload["config"]["model"])).to(device).eval()
    CheckpointManager.load(args.checkpoint, model, device=device)
    model.eval()

    split_paths = [ROOT / "data" / "splits" / name for name in args.splits]
    rows: list[BrighteningRow] = load_brightening_rows(ROOT, split_paths)
    if args.limit_per_dataset > 0:
        rows = select_evenly(rows, args.limit_per_dataset)
    unique: dict[str, BrighteningRow] = {}
    for row in rows:
        unique.setdefault(row.image_id, row)
    selected = sorted(unique.values(), key=lambda row: row.image_id)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.cache_dir / "cache_manifest.json"
    checkpoint_sha = sha256(args.checkpoint)
    if manifest_path.exists() and not args.force:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("checkpoint_sha256") != checkpoint_sha:
            raise SystemExit(
                "pipeline source cache was built from a different checkpoint; "
                "use --force to rebuild it"
            )

    print(
        f"images={len(selected)} cache={args.cache_dir} "
        f"checkpoint={args.checkpoint} device={device}"
    )
    started = time.perf_counter()
    entries = []
    computed = 0
    skipped = 0
    for start in range(0, len(selected), args.batch_size):
        batch = selected[start : start + args.batch_size]
        pending: list[tuple[BrighteningRow, Path]] = []
        sources = []
        for row in batch:
            path = source_cache_path(args.cache_dir, row.image_id)
            if path.exists() and not args.force:
                skipped += 1
                entries.append(
                    {
                        "image_id": row.image_id,
                        "dataset": row.dataset,
                        "split": row.split,
                        "cache_file": path.name,
                        "sha256": sha256(path),
                    }
                )
                continue
            pending.append((row, path))
            sources.append(load_rgb(row.low_path))
        if pending:
            outputs = predict_batch(model, sources, device)
            for (row, path), output in zip(pending, outputs, strict=True):
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, output.astype(np.float16))
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
        done = min(start + args.batch_size, len(selected))
        if computed and done % 40 == 0:
            print(
                f"  {done}/{len(selected)} computed={computed} skipped={skipped} "
                f"elapsed={time.perf_counter() - started:.0f}s"
            )

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "denoiser": "cnn",
        "checkpoint": str(args.checkpoint.relative_to(ROOT))
        if args.checkpoint.is_relative_to(ROOT)
        else str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "dtype": "float16",
        "n_images": len(entries),
        "computed": computed,
        "skipped": skipped,
        "entries": entries,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"manifest={manifest_path} "
        f"computed={computed} skipped={skipped} "
        f"seconds={time.perf_counter() - started:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
