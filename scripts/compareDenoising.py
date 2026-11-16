"""Compare Stage 03 CNN predictions with frozen classical denoisers."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lowlight.utils.io import read_csv_rows, write_csv_rows  # noqa: E402

METRICS = ("psnr", "ssim", "lpips", "niqe")
LEVELS = ("clean", "low", "medium", "high")
HIGHER_BETTER = {"psnr", "ssim"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cnn",
        type=Path,
        default=ROOT / "results" / "denoising_cnn" / "evaluation" / "per_image.csv",
    )
    parser.add_argument(
        "--d-fixed",
        type=Path,
        default=ROOT / "results" / "denoising_cnn" / "evaluation_d_fixed" / "per_image.csv",
    )
    parser.add_argument(
        "--classical",
        type=Path,
        default=ROOT / "results" / "classical_denoising" / "per_image.csv",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "denoising_cnn")
    parser.add_argument("--resamples", type=int, default=1000)
    return parser.parse_args()


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["dataset"], row["image_id"], row["noise_level"]


def bootstrap_ci(values: np.ndarray, resamples: int, seed: int = 0) -> tuple[float, float]:
    if values.size < 2:
        return float(values.mean()), float(values.mean())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda index: pvalues[index])
    adjusted = [1.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(pvalues) - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def paired_test(differences: np.ndarray) -> tuple[str, float, float, str]:
    differences = differences[np.isfinite(differences)]
    if differences.size < 3 or np.allclose(differences, 0.0):
        return "none", 1.0, 0.0, "none"
    p_normal = stats.shapiro(differences).pvalue if differences.size >= 3 else 0.0
    if p_normal > 0.05:
        result = stats.ttest_rel(differences, np.zeros_like(differences))
        effect = (
            float(differences.mean() / differences.std(ddof=1)) if differences.std(ddof=1) else 0.0
        )
        return "paired_t", float(result.pvalue), effect, "cohens_dz"
    result = stats.wilcoxon(differences, zero_method="wilcox")
    positives = differences[differences > 0].sum()
    negatives = -differences[differences < 0].sum()
    effect = (
        float((positives - negatives) / (positives + negatives)) if positives + negatives else 0.0
    )
    return "wilcoxon", float(result.pvalue), effect, "rank_biserial"


def main() -> int:
    args = parse_args()
    cnn_rows = read_csv_rows(args.cnn)
    dfixed_rows = read_csv_rows(args.d_fixed)
    classical_rows = read_csv_rows(args.classical)

    cnn_by_seed: dict[tuple[str, str, str], dict[int, dict[str, float]]] = defaultdict(dict)
    for row in cnn_rows:
        cnn_by_seed[key(row)][int(row["seed"])] = {metric: float(row[metric]) for metric in METRICS}
    cnn_mean: dict[tuple[str, str, str], dict[str, float]] = {}
    cnn_seed_std: dict[tuple[str, str, str], dict[str, float]] = {}
    for item_key, seeds in cnn_by_seed.items():
        values = np.asarray([[seeds[seed][metric] for metric in METRICS] for seed in sorted(seeds)])
        cnn_mean[item_key] = {
            metric: float(values[:, index].mean()) for index, metric in enumerate(METRICS)
        }
        cnn_seed_std[item_key] = {
            metric: float(values[:, index].std(ddof=1)) if values.shape[0] > 1 else 0.0
            for index, metric in enumerate(METRICS)
        }

    dfixed = {key(row): {metric: float(row[metric]) for metric in METRICS} for row in dfixed_rows}
    classical: dict[tuple[str, str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    for row in classical_rows:
        classical[key(row)][row["method"]] = {metric: float(row[metric]) for metric in METRICS}
    classical_psnr = defaultdict(list)
    for (dataset, _, level), methods in classical.items():
        for method, values in methods.items():
            classical_psnr[(dataset, level, method)].append(values["psnr"])
    best_classical = {}
    for dataset, _, level in classical:
        candidates = [
            (float(np.mean(values)), method)
            for (data, noise, method), values in classical_psnr.items()
            if data == dataset and noise == level
        ]
        best_classical[(dataset, level)] = max(candidates)[1]

    comparison_rows = []
    for (dataset, image_id, level), candidate in cnn_mean.items():
        references = {"d_fixed": dfixed[(dataset, image_id, level)]}
        best_method = best_classical[(dataset, level)]
        references[f"classical:{best_method}"] = classical[(dataset, image_id, level)][best_method]
        for reference_name, reference in references.items():
            for metric in METRICS:
                delta = candidate[metric] - reference[metric]
                comparison_rows.append(
                    {
                        "dataset": dataset,
                        "noise_level": level,
                        "candidate": "cnn_unet",
                        "reference": reference_name,
                        "metric": metric,
                        "n_images": 0,
                        "n_seeds": len(cnn_by_seed[(dataset, image_id, level)]),
                        "candidate_mean": candidate[metric],
                        "candidate_seed_std": cnn_seed_std[(dataset, image_id, level)][metric],
                        "reference_mean": reference[metric],
                        "delta": delta,
                    }
                )

    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in comparison_rows:
        grouped[(row["dataset"], row["noise_level"], row["reference"], row["metric"])].append(row)
    table = []
    for (dataset, level, reference, metric), rows in grouped.items():
        deltas = np.asarray([float(row["delta"]) for row in rows])
        candidate_values = np.asarray([float(row["candidate_mean"]) for row in rows])
        reference_values = np.asarray([float(row["reference_mean"]) for row in rows])
        ci_low, ci_high = bootstrap_ci(deltas, args.resamples)
        test, pvalue, effect, effect_type = paired_test(deltas)
        favorable = deltas.mean() > 0 if metric in HIGHER_BETTER else deltas.mean() < 0
        table.append(
            {
                "dataset": dataset,
                "noise_level": level,
                "candidate": "cnn_unet",
                "reference": reference,
                "metric": metric,
                "n_images": len(rows),
                "n_seeds": max(int(row["n_seeds"]) for row in rows),
                "candidate_mean": float(candidate_values.mean()),
                "candidate_std_images": float(candidate_values.std(ddof=1)),
                "candidate_seed_std": float(
                    np.mean([float(row["candidate_seed_std"]) for row in rows])
                ),
                "reference_mean": float(reference_values.mean()),
                "delta": float(deltas.mean()),
                "delta_ci95_low": ci_low,
                "delta_ci95_high": ci_high,
                "favorable": favorable,
                "test": test,
                "p_value": pvalue,
                "effect_size": effect,
                "effect_type": effect_type,
            }
        )
    adjusted = holm([float(row["p_value"]) for row in table])
    for row, value in zip(table, adjusted, strict=True):
        row["p_value_holm"] = value
        row["significant_holm"] = value < 0.05

    columns = [
        "dataset",
        "noise_level",
        "candidate",
        "reference",
        "metric",
        "n_images",
        "n_seeds",
        "candidate_mean",
        "candidate_std_images",
        "candidate_seed_std",
        "reference_mean",
        "delta",
        "delta_ci95_low",
        "delta_ci95_high",
        "favorable",
        "test",
        "p_value",
        "p_value_holm",
        "significant_holm",
        "effect_size",
        "effect_type",
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(args.out_dir / "comparison.csv", table, columns)
    summary = {
        "candidate": "cnn_unet",
        "n_images": len({(row["dataset"], row["image_id"]) for row in cnn_rows}),
        "n_seeds": len({int(row["seed"]) for row in cnn_rows}),
        "best_classical": {
            f"{dataset}|{level}": method for (dataset, level), method in best_classical.items()
        },
        "statistics": {
            "paired_test": "Shapiro normality; paired t-test if p>0.05 else Wilcoxon",
            "multiple_comparison": "Holm-Bonferroni across all dataset/level/reference/metric rows",
            "bootstrap_resamples": args.resamples,
            "effect_size": "Cohen's dz for t-test; rank-biserial for Wilcoxon",
        },
    }
    (args.out_dir / "comparison_metadata.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"comparison={args.out_dir / 'comparison.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
