"""Build dataset manifests and scene grouping metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lowlight.data.auditor import audit_all  # noqa: E402
from lowlight.data.grouping import assign_scene_ids  # noqa: E402
from lowlight.data.manifest import (  # noqa: E402
    load_audit,
    load_dataset_versions,
    rows_from_report,
    write_checksums,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument(
        "--audit", type=Path, default=ROOT / "results" / "audit" / "data_audit.json"
    )
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data" / "manifests")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--near-dup-threshold", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.audit.exists():
        report = load_audit(args.audit)
    else:
        report = audit_all(args.raw_root, args.near_dup_threshold)
        args.audit.parent.mkdir(parents=True, exist_ok=True)
        args.audit.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    versions = load_dataset_versions(args.manifest_dir / "dataset_versions.json")
    groupings = {}
    grouping_report = {}
    for dataset, data in report.items():
        grouping = assign_scene_ids(data["pairs"], threshold=args.near_dup_threshold)
        groupings[dataset] = grouping
        grouping_report[dataset] = {
            "threshold": grouping.threshold,
            "basis": "GT SHA256 + GT pHash Hamming",
            "n_pairs": grouping.n_pairs,
            "n_groups": grouping.n_groups,
            "largest_group": grouping.largest_group,
            "groups": {
                scene_id: [f"{split}/{image_id}" for split, image_id in groups]
                for scene_id, groups in grouping.groups.items()
            },
        }

    rows = rows_from_report(report, groupings, versions, ROOT)

    all_rows = []
    manifest_paths = []
    for dataset in sorted(report):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        name = "lolv2_real" if dataset == "lolv2-real" else "lol"
        path = args.manifest_dir / f"{name}.csv"
        write_csv(path, dataset_rows)
        manifest_paths.append(path)
        all_rows.extend(dataset_rows)
        print(f"{dataset}: {len(dataset_rows)} pairs -> {path}")

    combined = args.manifest_dir / "combined.csv"
    write_csv(combined, all_rows)
    manifest_paths.append(combined)

    args.splits_dir.mkdir(parents=True, exist_ok=True)
    (args.splits_dir / "scene_grouping.json").write_text(
        json.dumps(grouping_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    versions_path = args.manifest_dir / "dataset_versions.json"
    if versions_path.exists():
        manifest_paths.append(versions_path)
    write_checksums(manifest_paths, args.manifest_dir / "checksums.sha256", ROOT)

    print(f"combined: {len(all_rows)} pairs -> {combined}")
    print(f"scene grouping -> {args.splits_dir / 'scene_grouping.json'}")
    print(f"checksums -> {args.manifest_dir / 'checksums.sha256'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
