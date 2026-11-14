"""Download pinned datasets from Hugging Face.

Examples:
    python scripts/downloadData.py --dataset lol
    python scripts/downloadData.py --dataset lolv2-real
    python scripts/downloadData.py --dataset lolv2-synthetic --backup-only
    python scripts/downloadData.py --check
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _relative_path(path: str) -> str:
    candidate = Path(path)
    try:
        return candidate.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return candidate.as_posix()

from lowlight.data.downloader import (  # noqa: E402
    DATASETS,
    HfDatasetDownloader,
    record_backup_only,
    update_dataset_versions,
    validate_revision,
    verify_downloads,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        help="dataset to download",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="40-character commit hash (defaults to the pinned revision)",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=ROOT / "data" / "raw",
        help="destination for raw data",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "manifests" / "dataset_versions.json",
        help="dataset_versions.json path",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    parser.add_argument(
        "--backup-only",
        action="store_true",
        help="record metadata for a backup dataset without downloading",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify existing downloads and hashes",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    if args.check:
        problems = verify_downloads(args.manifest, args.raw_root)
        if problems:
            for problem in problems:
                print(f"FAIL {problem}")
            return 1
        print("OK downloads and hashes verified")
        return 0

    if not args.dataset:
        print("error: --dataset is required unless --check is used", file=sys.stderr)
        return 2

    if args.revision is not None:
        validate_revision(args.revision)

    spec = DATASETS[args.dataset]
    if args.revision is not None:
        spec = replace(spec, revision=args.revision)

    if spec.status == "backup":
        if not args.backup_only:
            print(
                f"error: {spec.name} is backup only; pass --backup-only",
                file=sys.stderr,
            )
            return 2
        data = record_backup_only(spec, args.manifest)
        print(json.dumps(data[spec.name], indent=2))
        return 0

    downloader = HfDatasetDownloader(spec)
    result = downloader.fetcher(args.raw_root, force=args.force)
    result.path = _relative_path(result.path)
    update_dataset_versions(args.manifest, result)
    print(json.dumps(result.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
