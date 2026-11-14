from __future__ import annotations

from pathlib import Path

from lowlight.data.auditor import build_pairs, default_specs

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"


def test_pairs_are_complete_for_all_splits() -> None:
    if not RAW.exists():
        import pytest

        pytest.skip("raw data not downloaded")
    for spec in default_specs(RAW):
        for split_spec in spec.splits:
            records, orphan_low, orphan_high = build_pairs(spec.name, split_spec)
            assert records, f"{spec.name}/{split_spec.split} has no pairs"
            assert not orphan_low, f"{spec.name}/{split_spec.split} orphan low images"
            assert not orphan_high, f"{spec.name}/{split_spec.split} orphan high images"
            assert len(records) == spec.official_counts[split_spec.split]


def test_pairs_have_matching_names_and_dimensions() -> None:
    if not RAW.exists():
        import pytest

        pytest.skip("raw data not downloaded")
    spec = default_specs(RAW)[0]
    records, _, _ = build_pairs(spec.name, spec.splits[0])
    for record in records:
        assert Path(record.low_path).stem == record.image_id
        assert Path(record.high_path).stem == record.image_id
        assert (record.width, record.height) == spec.expected_dims
        assert record.error is None
