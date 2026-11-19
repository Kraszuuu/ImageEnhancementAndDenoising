"""Stage 07 analysis over existing per-image result files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lowlight.evaluation import (  # noqa: E402
    METRIC_COLUMNS,
    SourceSpec,
    aggregate_rows,
    analyze_rows,
    filter_rows,
    holm_correction,
    load_sources,
    paired_test,
    per_image_mean,
)
from lowlight.utils.io import write_csv_rows  # noqa: E402

SUMMARY_COLUMNS = [
    "metric",
    "mean",
    "std",
    "median",
    "ci95_low",
    "ci95_high",
    "seed_mean",
    "seed_std",
    "n",
    "n_seeds",
]
TEST_COLUMNS = [
    "comparison",
    "task",
    "condition",
    "dataset",
    "method_a",
    "method_b",
    "metric",
    "n",
    "mean_a",
    "mean_b",
    "mean_delta",
    "median_delta",
    "cohens_dz",
    "rank_biserial",
    "t_p_value",
    "t_p_holm",
    "wilcoxon_p_value",
    "wilcoxon_p_holm",
]
FAILURE_COLUMNS = [
    "source",
    "task",
    "method",
    "track",
    "dataset",
    "noise_level",
    "seed",
    "image_id",
    "psnr",
    "ssim",
    "lpips",
    "niqe",
    "brightness_error",
    "color_cast",
    "clip_low_fraction",
    "clip_high_fraction",
]


def _format_cell(value: object) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0.0 else "-inf"
        return f"{value:.6f}"
    return str(value)


def write_table(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    write_csv_rows(
        path,
        [{column: _format_cell(row.get(column, "")) for column in columns} for row in rows],
        columns,
    )


def resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def source_specs(config: dict[str, object]) -> list[SourceSpec]:
    specs = []
    raw_sources = config.get("sources", [])
    if not isinstance(raw_sources, list):
        raise ValueError("sources must be a list")
    for source in raw_sources:
        if not isinstance(source, dict):
            raise ValueError("each source must be a mapping")
        task = str(source["task"])
        default_noise = source.get("default_noise_level")
        if default_noise is None:
            default_noise = "real" if task == "brightening" else ""
        specs.append(
            SourceSpec(
                name=str(source["name"]),
                path=resolve(source["path"]),
                task=task,
                variant=str(source["variant"]),
                method_column=str(source.get("method_column", "method")),
                default_noise_level=str(default_noise),
                default_track=str(source.get("default_track", "")),
            )
        )
    return specs


def load_config(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("ablation config must be a mapping")
    return config


def summarize(
    rows: list[dict[str, str]],
    config: dict[str, object],
    output_dir: Path,
) -> None:
    bootstrap = config.get("bootstrap", {})
    if not isinstance(bootstrap, dict):
        raise ValueError("bootstrap must be a mapping")
    resamples = int(bootstrap.get("resamples", 1000))
    ci = float(bootstrap.get("ci", 0.95))
    seed = int(bootstrap.get("seed", 0))
    raw_summaries = config.get("summaries", [])
    if not isinstance(raw_summaries, list):
        raise ValueError("summaries must be a list")
    for summary in raw_summaries:
        if not isinstance(summary, dict):
            raise ValueError("each summary must be a mapping")
        group_by = [str(key) for key in summary["group_by"]]
        task = str(summary["task"])
        subset = filter_rows(rows, task=task)
        if not subset:
            continue
        table = aggregate_rows(
            subset,
            group_by,
            resamples=resamples,
            ci=ci,
            seed=seed,
        )
        write_table(
            output_dir / f"summary_{summary['name']}.csv",
            table,
            group_by + SUMMARY_COLUMNS,
        )


def _comparison_rows(
    rows: list[dict[str, str]],
    comparison: dict[str, object],
    metric: str,
    dataset: str | None,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    task_a = str(comparison.get("task_a", comparison["task"]))
    task_b = str(comparison.get("task_b", comparison["task"]))
    common = {
        "dataset": dataset,
        "noise_level": None
        if comparison.get("noise_level") in (None, "")
        else str(comparison["noise_level"]),
        "track": None if comparison.get("track") in (None, "") else str(comparison["track"]),
    }
    side_a = filter_rows(
        rows,
        **common,
        task=task_a,
        method=str(comparison["method_a"]),
        variant=None if comparison.get("variant_a") in (None, "") else str(comparison["variant_a"]),
    )
    side_b = filter_rows(
        rows,
        **common,
        task=task_b,
        method=str(comparison["method_b"]),
        variant=None if comparison.get("variant_b") in (None, "") else str(comparison["variant_b"]),
    )
    return per_image_mean(side_a, metric), per_image_mean(side_b, metric)


def compare(rows: list[dict[str, str]], config: dict[str, object], output_dir: Path) -> None:
    raw_comparisons = config.get("comparisons", [])
    if not isinstance(raw_comparisons, list):
        raise ValueError("comparisons must be a list")
    raw_datasets = config.get("comparison_datasets", ["all"])
    if not isinstance(raw_datasets, list):
        raise ValueError("comparison_datasets must be a list")
    table: list[dict[str, object]] = []
    for dataset_name in raw_datasets:
        dataset = None if dataset_name in (None, "", "all") else str(dataset_name)
        for comparison in raw_comparisons:
            if not isinstance(comparison, dict):
                raise ValueError("each comparison must be a mapping")
            condition = "|".join(
                str(comparison.get(key, ""))
                for key in ("noise_level", "track")
                if comparison.get(key) not in (None, "")
            )
            for metric in METRIC_COLUMNS:
                values_a, values_b = _comparison_rows(rows, comparison, metric, dataset)
                keys = sorted(set(values_a) & set(values_b))
                if len(keys) < 2:
                    continue
                result = paired_test(
                    [values_a[key] for key in keys],
                    [values_b[key] for key in keys],
                )
                table.append(
                    {
                        "comparison": comparison["name"],
                        "task": comparison["task"],
                        "condition": condition,
                        "dataset": dataset_name,
                        "method_a": comparison["method_a"],
                        "method_b": comparison["method_b"],
                        "metric": metric,
                        "n": result.n,
                        "mean_a": result.mean_a,
                        "mean_b": result.mean_b,
                        "mean_delta": result.mean_delta,
                        "median_delta": result.median_delta,
                        "cohens_dz": result.cohens_dz,
                        "rank_biserial": result.rank_biserial,
                        "t_p_value": result.t_p_value,
                        "wilcoxon_p_value": result.wilcoxon_p_value,
                    }
                )

    for metric in METRIC_COLUMNS:
        for dataset_name in raw_datasets:
            subset = [
                row for row in table if row["metric"] == metric and row["dataset"] == dataset_name
            ]
            t_adjusted = holm_correction([float(row["t_p_value"]) for row in subset])
            w_adjusted = holm_correction([float(row["wilcoxon_p_value"]) for row in subset])
            for row, t_value, w_value in zip(subset, t_adjusted, w_adjusted, strict=True):
                row["t_p_holm"] = t_value
                row["wilcoxon_p_holm"] = w_value
    write_table(output_dir / "paired_tests.csv", table, TEST_COLUMNS)


def write_failure_cases(rows: list[dict[str, str]], output_dir: Path) -> None:
    pipeline = filter_rows(rows, task="pipeline")
    cases = analyze_rows(pipeline, metric="psnr", worst=True, limit=30)
    write_table(output_dir / "failure_cases.csv", cases, FAILURE_COLUMNS)

    brightening = filter_rows(rows, task="brightening", variant="cnn")
    metrics = [
        "brightness_error",
        "color_cast",
        "clip_low_fraction",
        "clip_high_fraction",
    ]
    summary = aggregate_rows(
        brightening,
        ["method", "track", "dataset"],
        metrics=metrics,
    )
    write_table(
        output_dir / "failure_metrics_by_method.csv",
        summary,
        ["method", "track", "dataset"] + SUMMARY_COLUMNS,
    )


def write_ablation_status(config: dict[str, object], output_dir: Path) -> None:
    raw_ablations = config.get("ablations", [])
    if not isinstance(raw_ablations, list):
        raise ValueError("ablations must be a list")
    rows = []
    for ablation in raw_ablations:
        if not isinstance(ablation, dict):
            raise ValueError("each ablation must be a mapping")
        rows.append(
            {
                "id": ablation["id"],
                "status": ablation["status"],
                "training_runs": ablation.get("training_runs", ""),
                "seeds": ",".join(str(seed) for seed in ablation.get("seeds", [])),
                "primary_metric": ablation.get("primary_metric", ""),
                "reason": ablation.get("reason", ""),
            }
        )
    columns = ["id", "status", "training_runs", "seeds", "primary_metric", "reason"]
    write_table(output_dir / "ablation_status.csv", rows, columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/ablations/ablations.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/ablations")
    args = parser.parse_args()

    config_path = resolve(args.config)
    output_dir = resolve(args.output_dir)
    config = load_config(config_path)
    rows = load_sources(source_specs(config))
    output_dir.mkdir(parents=True, exist_ok=True)

    write_table(
        output_dir / "unified_per_image.csv",
        rows,
        [
            "source",
            "task",
            "variant",
            "method",
            "track",
            "dataset",
            "split",
            "noise_level",
            "seed",
            "image_id",
            "psnr",
            "ssim",
            "lpips",
            "niqe",
            "inference_ms",
            "brightness_error",
            "color_cast",
            "clip_low_fraction",
            "clip_high_fraction",
        ],
    )
    summarize(rows, config, output_dir)
    compare(rows, config, output_dir)
    write_failure_cases(rows, output_dir)
    write_ablation_status(config, output_dir)

    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "n_rows": len(rows),
        "n_sources": len(config.get("sources", [])),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(rows)} rows to {output_dir}")


if __name__ == "__main__":
    main()
