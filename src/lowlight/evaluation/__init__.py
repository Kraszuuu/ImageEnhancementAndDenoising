"""Evaluation, aggregation and statistics."""

from lowlight.evaluation.aggregation import (
    FAILURE_COLUMNS,
    METRIC_COLUMNS,
    UNIFIED_COLUMNS,
    SourceSpec,
    aggregate_rows,
    filter_rows,
    load_source,
    load_sources,
    per_image_mean,
)
from lowlight.evaluation.failure_analysis import analyze_rows, failure_metrics
from lowlight.evaluation.statistics import (
    PairedTestResult,
    bootstrap_ci,
    holm_correction,
    paired_test,
)

__all__ = [
    "FAILURE_COLUMNS",
    "METRIC_COLUMNS",
    "PairedTestResult",
    "SourceSpec",
    "UNIFIED_COLUMNS",
    "aggregate_rows",
    "analyze_rows",
    "bootstrap_ci",
    "failure_metrics",
    "filter_rows",
    "holm_correction",
    "load_source",
    "load_sources",
    "paired_test",
    "per_image_mean",
]
