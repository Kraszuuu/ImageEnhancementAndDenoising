"""Evaluate frozen classical brighteners on the locked test splits (run once)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lowlight.classical.brighteners import build_brightener  # noqa: E402
from lowlight.classical.denoisers import build_denoiser  # noqa: E402
from lowlight.metrics import (  # noqa: E402
    LpipsMetric,
    NiqeMetric,
    PsnrMetric,
    SsimMetric,
    resolve_device,
)
from lowlight.utils.io import load_rgb, read_csv_rows, write_csv_rows  # noqa: E402

TEST_SPLITS = ("lol_test.csv", "lolv2_test.csv")
METRIC_NAMES = ("psnr", "ssim", "lpips", "niqe")
PER_IMAGE_COLUMNS = [
    "run_id",
    "dataset",
    "split",
    "track",
    "method",
    "image_id",
    "psnr",
    "ssim",
    "lpips",
    "niqe",
    "inference_ms",
    "source_ms",
    "width",
    "height",
]
SUMMARY_COLUMNS = [
    "track",
    "method",
    "dataset",
    "split",
    "metric",
    "mean",
    "std",
    "median",
    "ci95_low",
    "ci95_high",
    "n",
]
TIMING_COLUMNS = [
    "track",
    "method",
    "dataset",
    "device",
    "mean_ms",
    "std_ms",
    "median_ms",
    "n",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen",
        type=Path,
        default=ROOT / "configs" / "classical" / "brightening_frozen.yaml",
    )
    parser.add_argument(
        "--d-fixed",
        type=Path,
        default=ROOT / "configs" / "pipelines" / "d_fixed.yaml",
    )
    parser.add_argument("--splits", nargs="*", default=list(TEST_SPLITS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "results" / "classical_brightening"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def format_cell(value: object) -> object:
    if isinstance(value, float):
        if not math.isfinite(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.6f}"
    return value


def bootstrap_ci(
    values: np.ndarray, resamples: int = 1000, ci: float = 0.95, seed: int = 0
) -> tuple[float, float]:
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    low, high = np.percentile(means, [alpha * 100.0, (1.0 - alpha) * 100.0])
    return float(low), float(high)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    datasets = sorted({str(row["dataset"]) for row in rows})
    for track in sorted({str(row["track"]) for row in rows}):
        track_rows = [row for row in rows if row["track"] == track]
        for method in sorted({str(row["method"]) for row in track_rows}):
            method_rows = [row for row in track_rows if row["method"] == method]
            for dataset in [*datasets, "all"]:
                subset = (
                    method_rows
                    if dataset == "all"
                    else [row for row in method_rows if row["dataset"] == dataset]
                )
                if not subset:
                    continue
                for metric in METRIC_NAMES:
                    values = np.asarray(
                        [
                            float(row[metric])
                            for row in subset
                            if math.isfinite(float(row[metric]))
                        ],
                        dtype=np.float64,
                    )
                    if values.size == 0:
                        continue
                    low, high = bootstrap_ci(values)
                    output.append(
                        {
                            "track": track,
                            "method": method,
                            "dataset": dataset,
                            "split": subset[0]["split"],
                            "metric": metric,
                            "mean": float(values.mean()),
                            "std": float(values.std(ddof=1))
                            if values.size > 1
                            else 0.0,
                            "median": float(np.median(values)),
                            "ci95_low": low,
                            "ci95_high": high,
                            "n": int(values.size),
                        }
                    )
    return output


def load_rows(splits: list[str], limit: int) -> list[dict[str, str]]:
    rows = []
    for name in splits:
        rows.extend(read_csv_rows(ROOT / "data" / "splits" / name))
    if limit > 0:
        rows = rows[:limit]
    return rows


def metric_values(
    metrics: dict[str, object], prediction: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    return {
        name: float(metric.compute(prediction, target))  # type: ignore[attr-defined]
        for name, metric in metrics.items()
    }


def select_examples(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not rows:
        return []
    loaded = [(row, float(load_rgb(ROOT / row["low_path"]).mean())) for row in rows]
    loaded.sort(key=lambda item: item[1])
    indices = sorted({len(loaded) // 6, len(loaded) // 2, 5 * len(loaded) // 6})
    return [loaded[index][0] for index in indices]


def save_examples(
    path: Path,
    examples: list[dict[str, object]],
    best_fixed: str,
    best_raw: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = ["L", "D_fixed(L)", "target", f"{best_fixed} on D_fixed", f"{best_raw} on L"]
    figure, axes = plt.subplots(
        len(examples), len(columns), figsize=(3.2 * len(columns), 2.4 * len(examples))
    )
    axes = np.atleast_2d(axes)
    for row_index, example in enumerate(examples):
        panels = [
            example["low"],
            example["fixed"],
            example["target"],
            example["fixed_output"],
            example["raw_output"],
        ]
        for column_index, (label, image) in enumerate(zip(columns, panels, strict=True)):
            axis = axes[row_index, column_index]
            axis.imshow(np.clip(image, 0.0, 1.0))
            axis.set_title(
                f"{example['dataset']} {label}",
                fontsize=7,
            )
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def save_detail(
    path: Path,
    example: dict[str, object],
    method: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = ["L", "D_fixed(L)", "target", f"{method} on D_fixed"]
    panels = [
        example["low"],
        example["fixed"],
        example["target"],
        example["fixed_output"],
    ]
    height, width = np.asarray(example["target"]).shape[:2]
    y0, y1 = height // 3, 2 * height // 3
    x0, x1 = width // 4, 3 * width // 4
    figure, axes = plt.subplots(2, len(columns), figsize=(3.2 * len(columns), 5.2))
    for row_index, (row_label, crop) in enumerate(
        [("full", False), ("crop", True)]
    ):
        for column_index, (label, image) in enumerate(zip(columns, panels, strict=True)):
            panel = np.asarray(image)
            if crop:
                panel = panel[y0:y1, x0:x1]
            axis = axes[row_index, column_index]
            axis.imshow(np.clip(panel, 0.0, 1.0))
            axis.set_title(f"{row_label} {label}", fontsize=7)
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_failure_cases(
    path: Path,
    failures: list[dict[str, object]],
    best_fixed: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = ["L", "D_fixed(L)", "target", f"{best_fixed} on D_fixed"]
    figure, axes = plt.subplots(
        len(failures), len(columns), figsize=(3.2 * len(columns), 2.4 * len(failures))
    )
    axes = np.atleast_2d(axes)
    for row_index, failure in enumerate(failures):
        panels = [
            failure["low"],
            failure["fixed"],
            failure["target"],
            failure["fixed_output"],
        ]
        for column_index, (label, image) in enumerate(zip(columns, panels, strict=True)):
            axis = axes[row_index, column_index]
            axis.imshow(np.clip(image, 0.0, 1.0))
            axis.set_title(
                f"{failure['dataset']} {label}",
                fontsize=7,
            )
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir
    per_image_path = out_dir / "per_image.csv"
    if args.limit > 0 and any(name in TEST_SPLITS for name in args.splits):
        raise SystemExit("--limit is forbidden on locked test splits")
    is_test = tuple(args.splits) == TEST_SPLITS and args.limit == 0
    if per_image_path.exists() and not args.force:
        raise SystemExit(
            f"refusing to overwrite {per_image_path}; test is locked "
            "(use --force only with a documented reason)"
        )

    frozen = yaml.safe_load(args.frozen.read_text(encoding="utf-8"))
    d_fixed_payload = yaml.safe_load(args.d_fixed.read_text(encoding="utf-8"))
    tracks = list(frozen["tracks"])
    methods = list(frozen["tracks"][tracks[0]]["methods"])
    for track in tracks:
        if list(frozen["tracks"][track]["methods"]) != methods:
            raise SystemExit("frozen tracks must expose the same methods")
    best_by_track = {
        track: str(frozen["tracks"][track]["best_method"]) for track in tracks
    }
    rows = load_rows(args.splits, args.limit)
    selected_examples = select_examples(rows)
    example_ids = {row["low_path"] for row in selected_examples}
    device = resolve_device(None if args.device == "auto" else args.device)
    metrics = {
        "psnr": PsnrMetric(),
        "ssim": SsimMetric(),
        "lpips": LpipsMetric(device=device),
        "niqe": NiqeMetric(device=device),
    }

    denoiser = build_denoiser(
        str(d_fixed_payload["name"]), dict(d_fixed_payload["params"])
    )
    brighteners = {
        (track, method): build_brightener(
            method, frozen["tracks"][track]["methods"][method]["params"]
        )
        for track in tracks
        for method in methods
    }

    print(f"images={len(rows)} tracks={tracks} methods={methods} device={device}")
    result_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    examples: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        low = load_rgb(ROOT / row["low_path"])
        target = load_rgb(ROOT / row["gt_path"])
        height, width = low.shape[:2]
        start = time.perf_counter()
        fixed = denoiser.denoise(low)
        fixed_ms = (time.perf_counter() - start) * 1000.0
        sources = {"d_fixed": fixed, "raw": low}
        source_times = {"d_fixed": fixed_ms, "raw": 0.0}
        predictions: dict[tuple[str, str], np.ndarray] = {}
        for track in tracks:
            predictions[(track, "none")] = sources[track]
            for method in methods:
                start = time.perf_counter()
                prediction = brighteners[(track, method)].enhance(sources[track])
                elapsed = (time.perf_counter() - start) * 1000.0
                predictions[(track, method)] = prediction
                scores = metric_values(metrics, prediction, target)
                result_rows.append(
                    {
                        "run_id": "stage04-classical-brightening",
                        "dataset": row["dataset"],
                        "split": row["split"],
                        "track": track,
                        "method": method,
                        "image_id": row["low_path"],
                        "psnr": scores["psnr"],
                        "ssim": scores["ssim"],
                        "lpips": scores["lpips"],
                        "niqe": scores["niqe"],
                        "inference_ms": elapsed,
                        "source_ms": source_times[track],
                        "width": width,
                        "height": height,
                    }
                )
            scores = metric_values(metrics, sources[track], target)
            result_rows.append(
                {
                    "run_id": "stage04-classical-brightening",
                    "dataset": row["dataset"],
                    "split": row["split"],
                    "track": track,
                    "method": "none",
                    "image_id": row["low_path"],
                    "psnr": scores["psnr"],
                    "ssim": scores["ssim"],
                    "lpips": scores["lpips"],
                    "niqe": scores["niqe"],
                    "inference_ms": 0.0,
                    "source_ms": source_times[track],
                    "width": width,
                    "height": height,
                }
            )

        if row["low_path"] in example_ids:
            examples.append(
                {
                    "dataset": row["dataset"],
                    "low": low,
                    "fixed": fixed,
                    "target": target,
                    "fixed_output": predictions[("d_fixed", best_by_track["d_fixed"])],
                    "raw_output": predictions[("raw", best_by_track["raw"])],
                }
            )
        if index % 10 == 0 or index == len(rows):
            print(f"  {index}/{len(rows)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(
        per_image_path,
        [{key: format_cell(value) for key, value in row.items()} for row in result_rows],
        PER_IMAGE_COLUMNS,
    )
    summary_rows = summarize(result_rows)
    write_csv_rows(
        out_dir / "summary.csv",
        [{key: format_cell(value) for key, value in row.items()} for row in summary_rows],
        SUMMARY_COLUMNS,
    )

    timing_rows = []
    for track in tracks:
        for method in ["none", *methods]:
            for dataset in sorted({str(row["dataset"]) for row in result_rows}):
                values = [
                    float(row["inference_ms"])
                    if method != "none"
                    else float(row["source_ms"])
                    for row in result_rows
                    if row["track"] == track
                    and row["method"] == method
                    and row["dataset"] == dataset
                ]
                if not values:
                    continue
                array = np.asarray(values)
                timing_rows.append(
                    {
                        "track": track,
                        "method": method,
                        "dataset": dataset,
                        "device": "cuda" if device.type == "cuda" else "cpu",
                        "mean_ms": float(array.mean()),
                        "std_ms": float(array.std(ddof=1)) if array.size > 1 else 0.0,
                        "median_ms": float(np.median(array)),
                        "n": int(array.size),
                    }
                )
    write_csv_rows(
        out_dir / "timing.csv",
        [{key: format_cell(value) for key, value in row.items()} for row in timing_rows],
        TIMING_COLUMNS,
    )

    split_manifest = json.loads(
        (ROOT / "data" / "splits" / "split_manifest.json").read_text(encoding="utf-8")
    )
    metadata = {
        "run_id": "stage04-classical-brightening",
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "device": str(device),
        "test_used": is_test,
        "splits": list(args.splits),
        "limit": args.limit,
        "n_images": len(rows),
        "n_rows": len(result_rows),
        "frozen_config": str(args.frozen.relative_to(ROOT)),
        "frozen_config_sha256": sha256(args.frozen),
        "d_fixed_config": str(args.d_fixed.relative_to(ROOT)),
        "d_fixed_config_sha256": sha256(args.d_fixed),
        "d_fixed": d_fixed_payload,
        "split_hashes": split_manifest.get("files_sha256", {}),
        "tracks": {
            track: {
                "best_method": best_by_track[track],
                "methods": {
                    method: frozen["tracks"][track]["methods"][method]["params"]
                    for method in methods
                },
                "baseline": "none",
            }
            for track in tracks
        },
        "metrics": [metric.metadata() for metric in metrics.values()],
        "aggregation": "bootstrap 1000, ci 0.95, seed 0; non-finite excluded",
    }
    (out_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if is_test:
        (out_dir / "test_lock.json").write_text(
            json.dumps(
                {
                    "used_at_utc": metadata["created_at_utc"],
                    "split_hashes": metadata["split_hashes"],
                    "per_image_sha256": sha256(per_image_path),
                    "summary_sha256": sha256(out_dir / "summary.csv"),
                    "timing_sha256": sha256(out_dir / "timing.csv"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    if not args.no_figures:
        figures = out_dir / "figures"
        figures.mkdir(exist_ok=True)
        if examples:
            save_examples(
                figures / "brightening_examples.png",
                examples,
                best_by_track["d_fixed"],
                best_by_track["raw"],
            )
            save_detail(
                figures / "brightening_detail.png",
                examples[len(examples) // 2],
                best_by_track["d_fixed"],
            )
        failures: list[dict[str, object]] = []
        best_fixed = best_by_track["d_fixed"]
        by_id = {row["low_path"]: row for row in rows}
        for dataset in sorted({str(row["dataset"]) for row in rows}):
            candidates = sorted(
                [
                    row
                    for row in result_rows
                    if row["track"] == "d_fixed"
                    and row["method"] == best_fixed
                    and row["dataset"] == dataset
                ],
                key=lambda row: float(row["psnr"]),
            )
            seen_scenes: set[str] = set()
            picked = 0
            for candidate in candidates:
                source_row = by_id[str(candidate["image_id"])]
                scene = str(source_row["scene_id"])
                if scene in seen_scenes:
                    continue
                seen_scenes.add(scene)
                low = load_rgb(ROOT / source_row["low_path"])
                target = load_rgb(ROOT / source_row["gt_path"])
                fixed = denoiser.denoise(low)
                output = brighteners[("d_fixed", best_fixed)].enhance(fixed)
                failures.append(
                    {
                        "dataset": dataset,
                        "low": low,
                        "fixed": fixed,
                        "target": target,
                        "fixed_output": output,
                    }
                )
                picked += 1
                if picked == 3:
                    break
        if failures:
            save_failure_cases(figures / "failure_cases.png", failures, best_fixed)
        print(f"figures={figures}")

    print(f"per_image={per_image_path}")
    print(f"summary={out_dir / 'summary.csv'}")
    print(f"timing={out_dir / 'timing.csv'}")
    print(f"metadata={out_dir / 'run_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
