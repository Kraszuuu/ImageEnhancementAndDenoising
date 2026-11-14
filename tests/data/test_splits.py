from __future__ import annotations

import hashlib
import json
from pathlib import Path

from lowlight.data.manifest import read_csv

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ROOT / "data" / "splits"


def _split_files() -> dict[str, list[dict[str, str]]]:
    result = {}
    for name in (
        "lol_train.csv",
        "lol_val.csv",
        "lol_test.csv",
        "lolv2_train.csv",
        "lolv2_val.csv",
        "lolv2_test.csv",
    ):
        path = SPLITS / name
        if not path.exists():
            import pytest

            pytest.skip(f"missing split: {path}")
        result[name] = read_csv(path)
    return result


def test_split_sizes_and_assignments() -> None:
    splits = _split_files()
    for name, rows in splits.items():
        expected_split = name.split("_")[1].split(".")[0]
        assert rows
        assert all(row["split"] == expected_split for row in rows)


def test_scene_disjoint_train_and_val() -> None:
    splits = _split_files()
    for prefix in ("lol", "lolv2"):
        train_scenes = {r["scene_id"] for r in splits[f"{prefix}_train.csv"]}
        val_scenes = {r["scene_id"] for r in splits[f"{prefix}_val.csv"]}
        assert not (train_scenes & val_scenes)


def test_no_exact_pair_leakage_into_test() -> None:
    splits = _split_files()
    for prefix in ("lol", "lolv2"):
        fit_rows = splits[f"{prefix}_train.csv"] + splits[f"{prefix}_val.csv"]
        fit_hashes = {
            digest
            for row in fit_rows
            for digest in (row["low_sha256"], row["gt_sha256"])
        }
        test_hashes = {
            digest
            for row in splits[f"{prefix}_test.csv"]
            for digest in (row["low_sha256"], row["gt_sha256"])
        }
        assert not (fit_hashes & test_hashes)


def test_test_split_is_official() -> None:
    splits = _split_files()
    manifest = read_csv(ROOT / "data" / "manifests" / "combined.csv")
    official_test = {(r["dataset"], r["low_path"]) for r in manifest if r["split"] == "test"}
    for prefix in ("lol", "lolv2"):
        rows = splits[f"{prefix}_test.csv"]
        actual = {(r["dataset"], r["low_path"]) for r in rows}
        assert actual == {
            item for item in official_test if item[0] == rows[0]["dataset"]
        }


def test_split_hashes_are_frozen() -> None:
    manifest = json.loads((SPLITS / "split_manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest["files_sha256"].items():
        path = SPLITS / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
