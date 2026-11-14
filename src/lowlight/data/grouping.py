"""Scene grouping for leak-free train/validation splits.

Datasets do not ship explicit scene identifiers. Pairs are grouped when:

1. their normal-light GT images are byte-identical (strong duplicate signal), or
2. their GT perceptual hashes differ by at most ``threshold`` bits.

GT (normal-light) images are used because low-light frames are dark and produce
degenerate perceptual hashes.
"""

from __future__ import annotations

from dataclasses import dataclass

from lowlight.data.hashing import hamming


@dataclass
class SceneGrouping:
    scene_ids: dict[tuple[str, str], str]
    groups: dict[str, list[tuple[str, str]]]
    threshold: int
    n_pairs: int
    n_groups: int
    largest_group: int


def assign_scene_ids(records: list[dict], threshold: int = 4) -> SceneGrouping:
    """Assign deterministic scene ids to records (dicts from the audit report)."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    keys = [(record["split"], record["image_id"]) for record in records]
    n = len(records)
    parent = list(range(n))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_sha: dict[str, int] = {}
    for index, record in enumerate(records):
        digest = record.get("high_sha256")
        if not digest:
            continue
        if digest in by_sha:
            union(index, by_sha[digest])
        else:
            by_sha[digest] = index

    for i in range(n):
        left = records[i].get("high_phash")
        if left is None:
            continue
        for j in range(i + 1, n):
            right = records[j].get("high_phash")
            if right is not None and hamming(left, right) <= threshold:
                union(i, j)

    clusters: dict[int, list[tuple[str, str]]] = {}
    for index, key in enumerate(keys):
        clusters.setdefault(find(index), []).append(key)

    ordered = sorted(clusters.values(), key=lambda group: min(group))
    first = records[0]["dataset"] if records else "dataset"
    scene_ids: dict[tuple[str, str], str] = {}
    groups: dict[str, list[tuple[str, str]]] = {}
    for position, group in enumerate(ordered, start=1):
        scene_id = f"{first}_s{position:04d}"
        groups[scene_id] = sorted(group)
        for key in group:
            scene_ids[key] = scene_id

    return SceneGrouping(
        scene_ids=scene_ids,
        groups=groups,
        threshold=threshold,
        n_pairs=n,
        n_groups=len(groups),
        largest_group=max((len(g) for g in groups.values()), default=0),
    )
