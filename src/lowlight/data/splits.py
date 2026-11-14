"""Frozen train/validation/test split construction."""

from __future__ import annotations

import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path

from lowlight.data.manifest import MANIFEST_COLUMNS, write_csv


def write_csv_rows(
    path: Path,
    rows: list[dict[str, object]],
    columns: list[str] = MANIFEST_COLUMNS,
) -> None:
    write_csv(path, rows, columns)


def split_train_val(
    train_rows: list[dict[str, object]],
    seed: int,
    val_fraction: float,
    min_val_pairs: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Split train rows by scene groups, keeping all scenes intact."""
    if not train_rows:
        raise ValueError("train_rows is empty")

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in train_rows:
        grouped.setdefault(str(row["scene_id"]), []).append(row)

    group_ids = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    target = max(min_val_pairs, round(val_fraction * len(train_rows)))
    val_ids: set[str] = set()
    val_count = 0
    for group_id in group_ids:
        if val_count >= target:
            break
        val_ids.add(group_id)
        val_count += len(grouped[group_id])

    if val_count < min_val_pairs:
        raise ValueError(
            f"unable to reach {min_val_pairs} validation pairs; got {val_count}"
        )
    if len(val_ids) == len(group_ids):
        raise ValueError("scene grouping left no training groups")

    train_out: list[dict[str, object]] = []
    val_out: list[dict[str, object]] = []
    for group_id, rows in grouped.items():
        for row in rows:
            copy = dict(row)
            copy["split"] = "val" if group_id in val_ids else "train"
            (val_out if group_id in val_ids else train_out).append(copy)
    def sort_key(row: dict[str, object]) -> tuple[str, str, str]:
        label = row.get("image_id") or row["low_path"]
        return (str(row["dataset"]), str(row["scene_id"]), str(label))

    train_out.sort(key=sort_key)
    val_out.sort(key=sort_key)
    return train_out, val_out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scene_overlap(train_rows: list[dict], test_rows: list[dict]) -> dict[str, int]:
    train_scenes = {str(r["scene_id"]) for r in train_rows}
    test_scenes = {str(r["scene_id"]) for r in test_rows}
    return {
        "train_val_scenes": len(train_scenes),
        "test_scenes": len(test_scenes),
        "shared_scenes": len(train_scenes & test_scenes),
    }


def build_split_outputs(
    rows: list[dict[str, object]],
    out_dir: Path,
    seed: int = 0,
    val_fraction: float = 0.18,
    min_val_pairs: int = 30,
) -> dict[str, object]:
    """Build per-dataset split CSVs and the split manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = {}
    overlap: dict[str, dict[str, int]] = {}

    datasets = sorted({str(row["dataset"]) for row in rows})
    for dataset in datasets:
        dataset_rows = [r for r in rows if r["dataset"] == dataset]
        official_train = [r for r in dataset_rows if r["split"] == "train"]
        official_test = [r for r in dataset_rows if r["split"] == "test"]
        train_rows, val_rows = split_train_val(
            official_train, seed, val_fraction, min_val_pairs
        )
        test_rows = [dict(r, split="test") for r in official_test]

        prefix = "lolv2" if dataset == "lolv2-real" else "lol"
        for split_name, split_rows in (
            ("train", train_rows),
            ("val", val_rows),
            ("test", test_rows),
        ):
            path = out_dir / f"{prefix}_{split_name}.csv"
            write_csv_rows(path, split_rows)
            files[path.name] = _sha256(path)
        counts[dataset] = {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        }
        overlap[dataset] = _scene_overlap(train_rows + val_rows, test_rows)

    manifest = {
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "val_fraction": val_fraction,
        "min_val_pairs": min_val_pairs,
        "grouping": "GT SHA256 + GT pHash Hamming <= 4",
        "files_sha256": files,
        "counts": counts,
        "scene_overlap": overlap,
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
