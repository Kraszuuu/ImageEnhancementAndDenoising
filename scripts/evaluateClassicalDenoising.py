"""Evaluate frozen classical denoisers on the locked test splits (run once)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lowlight.classical.denoisers import build_denoiser  # noqa: E402
from lowlight.metrics import (  # noqa: E402
    LpipsMetric,
    NiqeMetric,
    PsnrMetric,
    SsimMetric,
    resolve_device,
    summarize,
)
from lowlight.metrics.runner import PER_IMAGE_COLUMNS, SUMMARY_COLUMNS  # noqa: E402
from lowlight.utils.io import load_rgb, read_csv_rows, write_csv_rows  # noqa: E402
from lowlight.utils.noise import load_noise_config  # noqa: E402

TEST_SPLITS = ("lol_test.csv", "lolv2_test.csv")
LEVELS = ("clean", "low", "medium", "high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frozen",
        type=Path,
        default=ROOT / "configs" / "classical" / "denoising_frozen.yaml",
    )
    parser.add_argument(
        "--noise-config", type=Path, default=ROOT / "configs" / "data" / "noise.yaml"
    )
    parser.add_argument("--splits", nargs="*", default=list(TEST_SPLITS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "results" / "classical_denoising"
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
    values = {}
    for name, metric in metrics.items():
        values[name] = float(metric.compute(prediction, target))  # type: ignore[attr-defined]
    return values


def save_figure(
    target: np.ndarray,
    noisy: dict[str, np.ndarray],
    outputs: dict[str, np.ndarray],
    path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("target L", target), *[(f"L+n {level}", noisy[level]) for level in noisy]]
    method_panels = [(method, image) for method, image in outputs.items()]
    count = len(panels) + len(method_panels)
    columns = 5
    rows = math.ceil(count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 3.0 * rows))
    for axis in axes.ravel():
        axis.axis("off")
    for axis, (label, image) in zip(axes.ravel(), panels + method_panels, strict=False):
        axis.imshow(np.clip(image, 0.0, 1.0))
        axis.set_title(label, fontsize=8)
        axis.axis("off")
    figure.suptitle(title, fontsize=10)
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
    noise_config = load_noise_config(args.noise_config)
    methods = list(frozen["methods"])
    rows = load_rows(args.splits, args.limit)
    device = resolve_device(None if args.device == "auto" else args.device)
    metrics = {
        "psnr": PsnrMetric(),
        "ssim": SsimMetric(),
        "lpips": LpipsMetric(device=device),
        "niqe": NiqeMetric(device=device),
    }

    denoisers = {
        (method, level): build_denoiser(method, frozen["methods"][method][level]["params"])
        for method in methods
        for level in LEVELS
    }

    print(f"images={len(rows)} methods={methods} levels={LEVELS} device={device}")
    result_rows: list[dict[str, object]] = []
    example: dict[str, object] | None = None
    example_id = None
    for index, row in enumerate(rows, start=1):
        clean = load_rgb(ROOT / row["low_path"])
        height, width = clean.shape[:2]
        if example_id is None:
            example_id = row["low_path"]
        noisy_by_level = {
            level: noise_config.add(clean, level, image_id=row["low_path"])
            for level in LEVELS
        }
        outputs: dict[str, np.ndarray] = {}
        for level in LEVELS:
            noisy = noisy_by_level[level]
            for method in methods:
                start = time.perf_counter()
                prediction = denoisers[(method, level)].denoise(noisy)
                elapsed = (time.perf_counter() - start) * 1000.0
                scores = metric_values(metrics, prediction, clean)
                result_rows.append(
                    {
                        "run_id": "stage02-classical",
                        "dataset": row["dataset"],
                        "split": row["split"],
                        "method": method,
                        "seed": 0,
                        "noise_level": level,
                        "image_id": row["low_path"],
                        "psnr": scores["psnr"],
                        "ssim": scores["ssim"],
                        "lpips": scores["lpips"],
                        "niqe": scores["niqe"],
                        "inference_ms": elapsed,
                        "width": width,
                        "height": height,
                    }
                )
                if row["low_path"] == example_id and level in {"clean", "high"}:
                    outputs[f"{method} {level}"] = prediction
        if row["low_path"] == example_id:
            example = {"target": clean, "noisy": noisy_by_level, "outputs": outputs}
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
    for method in methods:
        for dataset in sorted({str(row["dataset"]) for row in result_rows}):
            for level in LEVELS:
                values = [
                    float(row["inference_ms"])
                    for row in result_rows
                    if row["method"] == method
                    and row["dataset"] == dataset
                    and row["noise_level"] == level
                ]
                if not values:
                    continue
                array = np.asarray(values)
                timing_rows.append(
                    {
                        "method": method,
                        "dataset": dataset,
                        "device": "cpu",
                        "noise_level": level,
                        "mean_ms": float(array.mean()),
                        "std_ms": float(array.std(ddof=1)) if array.size > 1 else 0.0,
                        "median_ms": float(np.median(array)),
                        "n": int(array.size),
                    }
                )
    write_csv_rows(
        out_dir / "timing.csv",
        [{key: format_cell(value) for key, value in row.items()} for row in timing_rows],
        ["method", "dataset", "device", "noise_level", "mean_ms", "std_ms", "median_ms", "n"],
    )

    split_manifest = json.loads(
        (ROOT / "data" / "splits" / "split_manifest.json").read_text(encoding="utf-8")
    )
    d_fixed_path = ROOT / "configs" / "pipelines" / "d_fixed.yaml"
    metadata = {
        "run_id": "stage02-classical",
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "python": platform.python_version(),
        "device": str(device),
        "test_used": is_test,
        "splits": list(args.splits),
        "limit": args.limit,
        "n_images": len(rows),
        "n_rows": len(result_rows),
        "frozen_config": str(args.frozen.relative_to(ROOT)),
        "frozen_config_sha256": sha256(args.frozen),
        "noise_config": str(args.noise_config.relative_to(ROOT)),
        "noise_config_sha256": sha256(args.noise_config),
        "d_fixed": yaml.safe_load(d_fixed_path.read_text(encoding="utf-8"))
        if d_fixed_path.exists()
        else None,
        "d_fixed_sha256": sha256(d_fixed_path) if d_fixed_path.exists() else None,
        "split_hashes": split_manifest.get("files_sha256", {}),
        "methods": {
            method: {
                level: frozen["methods"][method][level]["params"]
                for level in LEVELS
            }
            for method in methods
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
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    if example is not None and not args.no_figures:
        figures = out_dir / "figures"
        figures.mkdir(exist_ok=True)
        save_figure(
            example["target"],
            {"high": example["noisy"]["high"]},
            {name: image for name, image in example["outputs"].items() if "high" in name},
            figures / "denoise_high.png",
            "Classical denoising, high noise",
        )
        save_figure(
            example["target"],
            {},
            {name: image for name, image in example["outputs"].items() if "clean" in name},
            figures / "denoise_clean.png",
            "Classical denoising, clean low-light",
        )
        print(f"figures={figures}")

    print(f"per_image={per_image_path}")
    print(f"summary={out_dir / 'summary.csv'}")
    print(f"timing={out_dir / 'timing.csv'}")
    print(f"metadata={out_dir / 'run_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
