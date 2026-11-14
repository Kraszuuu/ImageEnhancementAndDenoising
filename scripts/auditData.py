"""Audit downloaded datasets and write a Markdown report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lowlight.data.auditor import audit_all  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root", type=Path, default=ROOT / "data" / "raw", help="raw data root"
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=ROOT / "results" / "audit" / "data_audit.json",
        help="JSON report path",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=ROOT / "Outdated" / "Plans" / "handouts" / "stage00" / "DATA_AUDIT.md",
        help="Markdown report path",
    )
    parser.add_argument("--near-dup-threshold", type=int, default=8)
    return parser.parse_args()


def render_markdown(report: dict) -> str:
    lines: list[str] = ["# Data audit (Stage 00)", ""]
    for name, data in report.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"Root: `{data['root']}`")
        lines.append("")
        lines.append("### Counts")
        lines.append("")
        lines.append("| Split | Expected | Observed | Match |")
        lines.append("|---|---:|---:|:---:|")
        for split, check in data["count_checks"].items():
            lines.append(
                f"| {split} | {check['expected']} | {check['observed']} | "
                f"{'yes' if check['match'] else 'NO'} |"
            )
        lines.append("")
        dims = ", ".join(
            f"{d['width']}x{d['height']}" for d in data["dimensions"]
        ) or "n/a"
        lines.append(f"- Dimensions observed: {dims}")
        expected = data.get("expected_dimensions")
        if expected:
            lines.append(
                f"- Expected dimensions (width x height): "
                f"{expected['width']}x{expected['height']}"
            )
        lines.append(f"- Modes observed: {', '.join(data['modes']) or 'n/a'}")
        lines.append("")
        lines.append("### Integrity")
        lines.append("")
        lines.append(f"- Corrupt or mismatched pairs: {len(data['corrupt'])}")
        for item in data["corrupt"][:20]:
            lines.append(f"  - `{item['split']}/{item['image_id']}`: {item['error']}")
        lines.append(f"- Non-image files: {sum(data['non_image_files'].values())}")
        for split, orphans in data["orphans"].items():
            if orphans["low"] or orphans["high"]:
                lines.append(
                    f"- Orphans in `{split}`: low={len(orphans['low'])}, "
                    f"high={len(orphans['high'])}"
                )
        lines.append("")
        lines.append("### Duplicates and leakage")
        lines.append("")
        lines.append(
            f"- Exact duplicate groups (SHA256): {len(data['exact_duplicate_groups'])}"
        )
        for group in data["exact_duplicate_groups"][:10]:
            lines.append(f"  - {', '.join(group)}")
        lines.append(
            f"- Near-duplicate threshold (pHash Hamming): "
            f"{data['near_duplicate_threshold']}"
        )
        for split, clusters in data["near_duplicate_clusters"].items():
            sizes = sorted((len(c) for c in clusters), reverse=True)
            lines.append(
                f"- `{split}` scene clusters (GT pHash): {len(clusters)} "
                f"(largest: {sizes[0] if sizes else 0})"
            )
        leakage = data["leakage"]
        lines.append(
            f"- Test leakage (exact SHA256): {len(leakage['exact_sha_hits'])}"
        )
        lines.append(
            f"- Test leakage (pHash <= {leakage['phash_threshold']}): "
            f"{len(leakage['near_phash_hits'])}"
        )
        if leakage["near_phash_hits"]:
            for hit in leakage["near_phash_hits"][:10]:
                lines.append(
                    f"  - test/{hit['test_image']} hamming={hit['hamming']}"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    report = audit_all(args.raw_root, args.near_dup_threshold)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")

    failures = 0
    for name, data in report.items():
        for split, check in data["count_checks"].items():
            if not check["match"]:
                print(f"FAIL {name}/{split}: expected {check['expected']}, "
                      f"observed {check['observed']}")
                failures += 1
        if data["corrupt"]:
            print(f"FAIL {name}: {len(data['corrupt'])} corrupt pairs")
            failures += 1
        if data["leakage"]["exact_sha_hits"]:
            print(f"FAIL {name}: exact test leakage")
            failures += 1
        for split, orphans in data["orphans"].items():
            if orphans["low"] or orphans["high"]:
                print(f"FAIL {name}/{split}: orphan files")
                failures += 1
        if sum(data["non_image_files"].values()):
            print(f"FAIL {name}: non-image files present")
            failures += 1
        print(
            f"{name}: train={data['observed_counts'].get('train')} "
            f"test={data['observed_counts'].get('test')} "
            f"dimensions={data['dimensions']} modes={data['modes']}"
        )
    print(f"report: {args.json}")
    print(f"markdown: {args.markdown}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
