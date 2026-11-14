from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from lowlight.data.hashing import sha256_file
from lowlight.data.manifest import MANIFEST_COLUMNS, read_csv

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "data" / "manifests"


def _manifest(name: str) -> list[dict[str, str]]:
    path = MANIFESTS / name
    if not path.exists():
        import pytest

        pytest.skip(f"missing manifest: {path}")
    return read_csv(path)


def test_manifest_columns() -> None:
    rows = _manifest("combined.csv")
    assert rows
    assert list(rows[0].keys()) == MANIFEST_COLUMNS


def test_manifest_paths_and_hashes() -> None:
    rows = _manifest("combined.csv")
    for row in rows[:40]:
        low = ROOT / row["low_path"]
        gt = ROOT / row["gt_path"]
        assert low.is_file(), low
        assert gt.is_file(), gt
        assert not low.name.startswith("."), low
        assert sha256_file(low) == row["low_sha256"]
        assert sha256_file(gt) == row["gt_sha256"]
        with Image.open(low) as image:
            assert image.size == (int(row["width"]), int(row["height"]))
        with Image.open(gt) as image:
            assert image.size == (int(row["width"]), int(row["height"]))


def test_manifest_counts_and_revisions() -> None:
    rows = _manifest("combined.csv")
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["dataset"]] = counts.get(row["dataset"], 0) + 1
        assert len(row["source_revision"]) == 40
    assert counts == {"lol": 500, "lolv2-real": 789}


def test_dataset_versions_match_revisions() -> None:
    versions = json.loads((MANIFESTS / "dataset_versions.json").read_text(encoding="utf-8"))
    rows = _manifest("combined.csv")
    for row in rows[:40]:
        assert versions[row["dataset"]]["revision"] == row["source_revision"]
