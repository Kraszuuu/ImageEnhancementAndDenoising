"""Manifest building for paired low-light datasets."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from lowlight.data.grouping import SceneGrouping

MANIFEST_COLUMNS = [
    "dataset",
    "split",
    "scene_id",
    "low_path",
    "gt_path",
    "width",
    "height",
    "low_sha256",
    "gt_sha256",
    "source_revision",
]


def load_audit(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset_versions(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _relative(path: str, root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def rows_from_report(
    report: dict[str, object],
    groupings: dict[str, SceneGrouping],
    versions: dict[str, object],
    root: Path,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset, data in report.items():
        revision = versions.get(dataset, {}).get("revision", "")
        grouping = groupings[dataset]
        for record in data["pairs"]:
            key = (record["split"], record["image_id"])
            rows.append(
                {
                    "dataset": dataset,
                    "split": record["split"],
                    "scene_id": grouping.scene_ids[key],
                    "low_path": _relative(record["low_path"], root),
                    "gt_path": _relative(record["high_path"], root),
                    "width": record["width"],
                    "height": record["height"],
                    "low_sha256": record["low_sha256"],
                    "gt_sha256": record["high_sha256"],
                    "source_revision": revision,
                }
            )
    return rows


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    columns: list[str] = MANIFEST_COLUMNS,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_checksums(paths: list[Path], output: Path, root: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for path in sorted(paths):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.resolve().relative_to(root.resolve()).as_posix()}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
