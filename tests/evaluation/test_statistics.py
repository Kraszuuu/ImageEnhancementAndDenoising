from __future__ import annotations

import pytest

from lowlight.evaluation.statistics import bootstrap_ci, holm_correction, paired_test


def test_bootstrap_ci_constant_values() -> None:
    low, high = bootstrap_ci([2.0, 2.0, 2.0], resamples=100)
    assert low == pytest.approx(2.0)
    assert high == pytest.approx(2.0)


def test_bootstrap_ci_is_deterministic() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)


def test_holm_correction_preserves_order() -> None:
    adjusted = holm_correction([0.01, 0.02, 0.03])
    assert adjusted == pytest.approx([0.03, 0.04, 0.04])


def test_holm_correction_ignores_non_finite() -> None:
    adjusted = holm_correction([0.01, float("nan"), 0.04])
    assert adjusted[0] == pytest.approx(0.02)
    assert adjusted[2] == pytest.approx(0.04)


def test_paired_test_detects_positive_shift() -> None:
    result = paired_test([2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0])
    assert result.n == 4
    assert result.mean_delta == pytest.approx(1.0)
    assert result.cohens_dz > 1.0
    assert result.t_p_value < 0.05
    assert result.wilcoxon_p_value < 0.2


def test_paired_test_identical_samples() -> None:
    result = paired_test([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert result.mean_delta == pytest.approx(0.0)
    assert result.cohens_dz == pytest.approx(0.0)
    assert result.t_p_value == pytest.approx(1.0)
    assert result.wilcoxon_p_value == pytest.approx(1.0)


def test_paired_test_discards_non_finite_pairs() -> None:
    result = paired_test([1.0, float("nan"), 3.0], [0.0, 2.0, 2.0])
    assert result.n == 2
    assert result.mean_delta == pytest.approx(1.0)
