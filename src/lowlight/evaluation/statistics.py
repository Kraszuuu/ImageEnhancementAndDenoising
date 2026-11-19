"""Bootstrap intervals, paired tests and multiple-comparison correction."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class PairedTestResult:
    """Result of a two-sided paired comparison with effect sizes."""

    n: int
    mean_a: float
    mean_b: float
    mean_delta: float
    median_delta: float
    cohens_dz: float
    rank_biserial: float
    t_statistic: float
    t_p_value: float
    wilcoxon_statistic: float
    wilcoxon_p_value: float


def bootstrap_ci(
    values: Sequence[float],
    *,
    resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of the mean."""
    if resamples < 1:
        raise ValueError("resamples must be positive")
    if not 0.0 < ci < 1.0:
        raise ValueError("ci must be in (0, 1)")
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("bootstrap requires at least one finite value")
    if array.size == 1:
        value = float(array[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    means = array[indices].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    low, high = np.percentile(means, [alpha * 100.0, (1.0 - alpha) * 100.0])
    return float(low), float(high)


def holm_correction(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values, preserving input order."""
    values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.ones_like(values)
    finite = np.isfinite(values)
    if not finite.any():
        return adjusted.tolist()
    order = np.argsort(np.where(finite, values, np.inf))
    count = int(finite.sum())
    running = 0.0
    for position, index in enumerate(order[:count]):
        candidate = float(values[index]) * (count - position)
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def _rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0.0]
    if nonzero.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0.0].sum())
    return 2.0 * positive / (nonzero.size * (nonzero.size + 1.0)) - 1.0


def paired_test(a: Sequence[float], b: Sequence[float]) -> PairedTestResult:
    """Compare paired samples with a t-test and Wilcoxon signed-rank test.

    Non-finite pairs are discarded. ``a - b`` is the reported deltas, so a
    positive mean delta means that ``a`` is larger.
    """
    array_a = np.asarray(a, dtype=np.float64)
    array_b = np.asarray(b, dtype=np.float64)
    if array_a.shape != array_b.shape:
        raise ValueError("paired samples must have the same shape")
    finite = np.isfinite(array_a) & np.isfinite(array_b)
    array_a = array_a[finite]
    array_b = array_b[finite]
    if array_a.size < 2:
        raise ValueError("paired test requires at least two finite pairs")

    differences = array_a - array_b
    mean_delta = float(differences.mean())
    std_delta = float(differences.std(ddof=1))
    if std_delta == 0.0:
        cohens_dz = math.copysign(math.inf, mean_delta) if mean_delta else 0.0
    else:
        cohens_dz = mean_delta / std_delta

    if np.all(differences == 0.0):
        t_statistic = 0.0
        t_p_value = 1.0
        wilcoxon_statistic = 0.0
        wilcoxon_p_value = 1.0
    else:
        t_result = stats.ttest_rel(array_a, array_b)
        t_statistic = float(t_result.statistic)
        t_p_value = float(t_result.pvalue)
        if not math.isfinite(t_p_value):
            t_p_value = 1.0
        wilcoxon = stats.wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
        wilcoxon_statistic = float(wilcoxon.statistic)
        wilcoxon_p_value = float(wilcoxon.pvalue)

    return PairedTestResult(
        n=int(array_a.size),
        mean_a=float(array_a.mean()),
        mean_b=float(array_b.mean()),
        mean_delta=mean_delta,
        median_delta=float(np.median(differences)),
        cohens_dz=cohens_dz,
        rank_biserial=_rank_biserial(differences),
        t_statistic=t_statistic,
        t_p_value=t_p_value,
        wilcoxon_statistic=wilcoxon_statistic,
        wilcoxon_p_value=wilcoxon_p_value,
    )
