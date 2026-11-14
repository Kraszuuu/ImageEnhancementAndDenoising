"""Dataset auditing: pairing, dimensions, integrity, duplicates and leakage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from lowlight.data.hashing import hamming, perceptual_hash, sha256_file

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
NON_IMAGE_JUNK = {".ds_store", "thumbs.db"}


@dataclass
class PairRecord:
    dataset: str
    split: str
    image_id: str
    low_path: str
    high_path: str
    width: int | None = None
    height: int | None = None
    mode: str | None = None
    low_sha256: str | None = None
    high_sha256: str | None = None
    low_phash: int | None = None
    high_phash: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class SplitSpec:
    split: str
    low_dir: Path
    high_dir: Path


@dataclass(frozen=True)
class DatasetAuditSpec:
    name: str
    root: Path
    splits: tuple[SplitSpec, ...]
    expected_dims: tuple[int, int] | None
    official_counts: dict[str, int]


def list_images(directory: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    if not directory.is_dir():
        return result
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        if path.name.lower() in NON_IMAGE_JUNK:
            continue
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            result[path.stem] = path
    return result


def count_non_image_files(directory: Path) -> int:
    count = 0
    if not directory.is_dir():
        return count
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() not in IMAGE_EXTENSIONS:
            count += 1
    return count


def _relative(path: str, root: Path | None) -> str:
    if root is None:
        return Path(path).as_posix()
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _inspect(path: Path) -> tuple[
    int | None, int | None, str | None, str | None, int | None, str | None
]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
        return (
            width,
            height,
            mode,
            sha256_file(path),
            perceptual_hash(path),
            None,
        )
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return None, None, None, None, None, f"{type(exc).__name__}: {exc}"


def build_pairs(
    dataset: str, split_spec: SplitSpec
) -> tuple[list[PairRecord], list[str], list[str]]:
    low = list_images(split_spec.low_dir)
    high = list_images(split_spec.high_dir)
    orphan_low = sorted(set(low) - set(high))
    orphan_high = sorted(set(high) - set(low))
    records: list[PairRecord] = []
    for image_id in sorted(set(low) & set(high)):
        low_path = low[image_id]
        high_path = high[image_id]
        lw, lh, lmode, lsha, lphash, lerr = _inspect(low_path)
        hw, hh, hmode, hsha, hphash, herr = _inspect(high_path)
        error = lerr or herr
        if lerr is None and herr is None and (lw, lh) != (hw, hh):
            error = f"dimension mismatch: low={(lw, lh)} high={(hw, hh)}"
        records.append(
            PairRecord(
                dataset=dataset,
                split=split_spec.split,
                image_id=image_id,
                low_path=str(low_path.as_posix()),
                high_path=str(high_path.as_posix()),
                width=lw,
                height=lh,
                mode=lmode,
                low_sha256=lsha,
                high_sha256=hsha,
                low_phash=lphash,
                high_phash=hphash,
                error=error,
            )
        )
    return records, orphan_low, orphan_high


def _cluster_ids(values: list[tuple[str, int]], threshold: int) -> list[list[str]]:
    parent = list(range(len(values)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            if hamming(values[i][1], values[j][1]) <= threshold:
                union(i, j)

    groups: dict[int, list[str]] = {}
    for index, (image_id, _) in enumerate(values):
        groups.setdefault(find(index), []).append(image_id)
    return [sorted(group) for group in groups.values() if len(group) > 1]


def _exact_duplicate_groups(records: list[PairRecord]) -> list[list[str]]:
    seen: dict[str, list[str]] = {}
    for record in records:
        for digest in (record.low_sha256, record.high_sha256):
            if digest is not None:
                seen.setdefault(digest, []).append(f"{record.split}/{record.image_id}")
    groups = [sorted(set(group)) for group in seen.values() if len(set(group)) > 1]
    return sorted(groups)


def _leakage(
    records: list[PairRecord], threshold: int
) -> dict[str, object]:
    """Check test leakage using normal-light GT images.

    GT images are used instead of low-light images because pHash of very dark
    frames degenerates to near-identical values. GT is normal-light and far more
    discriminative for scene identity.
    """
    train = [r for r in records if r.split != "test"]
    test = [r for r in records if r.split == "test"]
    train_sha = {
        digest
        for r in train
        for digest in (r.low_sha256, r.high_sha256)
        if digest is not None
    }
    # Exact file-level leakage is checked on both low and GT bytes.
    sha_hits = [
        f"test/{r.image_id}"
        for r in test
        if (r.low_sha256 in train_sha) or (r.high_sha256 in train_sha)
    ]
    # Scene-level leakage is checked on GT pHash.
    train_phashes = [
        (r.image_id, r.high_phash) for r in train if r.high_phash is not None
    ]
    phash_hits: list[dict[str, object]] = []
    for record in test:
        if record.high_phash is None:
            continue
        nearest = min(
            (hamming(record.high_phash, other) for _, other in train_phashes),
            default=None,
        )
        if nearest is not None and nearest <= threshold:
            phash_hits.append({"test_image": record.image_id, "hamming": nearest})
    return {
        "exact_sha_hits": sha_hits,
        "near_phash_hits": phash_hits,
        "phash_threshold": threshold,
        "basis": "GT (high/normal-light)",
    }


def audit_dataset(
    spec: DatasetAuditSpec,
    near_dup_threshold: int = 8,
    path_root: Path | None = None,
) -> dict[str, object]:
    all_records: list[PairRecord] = []
    counts: dict[str, int] = {}
    orphans: dict[str, dict[str, list[str]]] = {}
    non_image: dict[str, int] = {}

    for split_spec in spec.splits:
        records, orphan_low, orphan_high = build_pairs(spec.name, split_spec)
        all_records.extend(records)
        counts[split_spec.split] = len(records)
        orphans[split_spec.split] = {
            "low": orphan_low,
            "high": orphan_high,
        }
        non_image[split_spec.split] = (
            count_non_image_files(split_spec.low_dir)
            + count_non_image_files(split_spec.high_dir)
        )

    for record in all_records:
        record.low_path = _relative(record.low_path, path_root)
        record.high_path = _relative(record.high_path, path_root)

    dims = sorted(
        {(r.width, r.height) for r in all_records if r.width and r.height}
    )
    modes = sorted({r.mode for r in all_records if r.mode})
    corrupt = [
        {"split": r.split, "image_id": r.image_id, "error": r.error}
        for r in all_records
        if r.error
    ]

    per_split_near: dict[str, list[list[str]]] = {}
    for split_spec in spec.splits:
        # Group by normal-light GT pHash: stable for dark frames and directly
        # reflects scene identity.
        values = [
            (r.image_id, r.high_phash)
            for r in all_records
            if r.split == split_spec.split and r.high_phash is not None
        ]
        per_split_near[split_spec.split] = _cluster_ids(values, near_dup_threshold)

    missing_counts = {
        split: {
            "expected": expected,
            "observed": counts.get(split),
            "match": counts.get(split) == expected,
        }
        for split, expected in spec.official_counts.items()
    }

    return {
        "dataset": spec.name,
        "root": _relative(str(spec.root), path_root),
        "official_counts": spec.official_counts,
        "observed_counts": counts,
        "count_checks": missing_counts,
        "dimensions": [{"width": w, "height": h} for w, h in dims],
        "expected_dimensions": (
            {"width": spec.expected_dims[0], "height": spec.expected_dims[1]}
            if spec.expected_dims
            else None
        ),
        "modes": modes,
        "corrupt": corrupt,
        "orphans": orphans,
        "non_image_files": non_image,
        "exact_duplicate_groups": _exact_duplicate_groups(all_records),
        "near_duplicate_threshold": near_dup_threshold,
        "near_duplicate_clusters": per_split_near,
        "leakage": _leakage(all_records, near_dup_threshold),
        "pairs": [record.__dict__ for record in all_records],
    }


def default_specs(raw_root: Path) -> list[DatasetAuditSpec]:
    lol_root = raw_root / "lol" / "extracted" / "lol_dataset"
    lolv2_root = raw_root / "lolv2-real"
    return [
        DatasetAuditSpec(
            name="lol",
            root=lol_root,
            splits=(
                SplitSpec("train", lol_root / "our485" / "low", lol_root / "our485" / "high"),
                SplitSpec("test", lol_root / "eval15" / "low", lol_root / "eval15" / "high"),
            ),
            expected_dims=(600, 400),
            official_counts={"train": 485, "test": 15},
        ),
        DatasetAuditSpec(
            name="lolv2-real",
            root=lolv2_root,
            splits=(
                SplitSpec("train", lolv2_root / "Train" / "Input", lolv2_root / "Train" / "GT"),
                SplitSpec("test", lolv2_root / "Test" / "Input", lolv2_root / "Test" / "GT"),
            ),
            expected_dims=(600, 400),
            official_counts={"train": 689, "test": 100},
        ),
    ]


def audit_all(
    raw_root: Path,
    near_dup_threshold: int = 8,
    path_root: Path | None = None,
) -> dict[str, object]:
    if path_root is None:
        path_root = raw_root.resolve().parent.parent
    return {
        spec.name: audit_dataset(spec, near_dup_threshold, path_root)
        for spec in default_specs(raw_root)
    }
