"""Evaluate frozen Stage 07 ablation checkpoints on the locked test splits (run once)."""

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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lowlight.metrics import (  # noqa: E402
    LpipsMetric,
    NiqeMetric,
    PsnrMetric,
    SsimMetric,
    resolve_device,
)
from lowlight.models import build_model, denoise_full_image  # noqa: E402
from lowlight.training.pipeline_datasets import darken_image  # noqa: E402
from lowlight.utils.io import load_rgb, read_csv_rows, write_csv_rows  # noqa: E402
from lowlight.utils.noise import load_noise_config  # noqa: E402

TEST_SPLITS = ("lol_test.csv", "lolv2_test.csv")
METRICS = ("psnr", "ssim", "lpips", "niqe")
LEVELS = ("real", "synthetic", "low", "medium", "high")
PER_IMAGE_COLUMNS = [
    "run_id",
    "group",
    "arm",
    "dataset",
    "split",
    "seed",
    "noise_level",
    "image_id",
    "psnr",
    "ssim",
    "lpips",
    "niqe",
    "inference_ms",
    "width",
    "height",
]

GROUP_SPECS: dict[str, dict[str, object]] = {
    "loss_l1": {
        "arms": {"l1_real": ("checkpoints/ablations/pilot/loss_l1/real", "real")},
        "levels": ("real",),
    },
    "composition": {
        "arms": {
            "lol_only": ("checkpoints/ablations/composition/lol_only/real", "real"),
            "lolv2_only": ("checkpoints/ablations/composition/lolv2_only/real", "real"),
        },
        "levels": ("real",),
    },
    "synthetic_brightening": {
        "arms": {
            "synthetic_brightening": (
                "checkpoints/ablations/synthetic/synthetic_brightening",
                "synthetic_brightening",
            )
        },
        "levels": ("synthetic",),
    },
    "synthetic_denoising": {
        "arms": {
            "synthetic_denoising": (
                "checkpoints/ablations/synthetic/synthetic_denoising",
                "synthetic_denoising",
            )
        },
        "levels": ("low", "medium", "high"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups", nargs="+", choices=tuple(GROUP_SPECS), default=list(GROUP_SPECS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--splits", nargs="*", default=list(TEST_SPLITS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--out-root", type=Path, default=ROOT / "results" / "ablations" / "evaluation"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def format_cell(value: object) -> object:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.6f}"
    return value


def load_rows(splits: list[str], limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
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


def batch_lpips(
    metric: LpipsMetric, predictions: list[np.ndarray], targets: list[np.ndarray], batch: int
) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(predictions), batch):
        pred = np.stack(
            [np.transpose(item, (2, 0, 1)) for item in predictions[start : start + batch]]
        ).astype(np.float32)
        ref = np.stack(
            [np.transpose(item, (2, 0, 1)) for item in targets[start : start + batch]]
        ).astype(np.float32)
        pred_t = torch.from_numpy(pred).to(metric.device) * 2.0 - 1.0
        ref_t = torch.from_numpy(ref).to(metric.device) * 2.0 - 1.0
        with torch.inference_mode():
            values = metric.model(pred_t, ref_t).reshape(-1).cpu().tolist()
        scores.extend(float(value) for value in values)
    return scores


def batch_niqe(metric: NiqeMetric, predictions: list[np.ndarray], batch: int) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(predictions), batch):
        array = np.stack(
            [np.transpose(item, (2, 0, 1)) for item in predictions[start : start + batch]]
        ).astype(np.float32)
        tensor = torch.from_numpy(array).to(metric.device)
        with torch.inference_mode():
            values = metric.model(tensor).reshape(-1).cpu().tolist()
        scores.extend(float(value) for value in values)
    return scores


def source_and_target(
    row: dict[str, str],
    *,
    image: np.ndarray,
    target: np.ndarray,
    mode: str,
    level: str,
    darken_gain: float,
    noise_config: object,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "real":
        return image, target
    darkened = darken_image(target, gain=darken_gain)
    if mode == "synthetic_brightening":
        return darkened, target
    if mode == "synthetic_denoising":
        return noise_config.add(darkened, level, image_id=row["low_path"], seed=0), darkened  # type: ignore[attr-defined]
    raise ValueError(f"unknown mode: {mode}")


def main() -> int:
    args = parse_args()
    if args.limit > 0 and tuple(args.splits) == TEST_SPLITS:
        raise SystemExit("--limit is forbidden on locked test splits")
    is_test = tuple(args.splits) == TEST_SPLITS and args.limit == 0
    if not args.out_root.exists():
        args.out_root.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.splits, args.limit)
    split_manifest = json.loads(
        (ROOT / "data" / "splits" / "split_manifest.json").read_text(encoding="utf-8")
    )
    device = resolve_device(None if args.device == "auto" else args.device)
    psnr = PsnrMetric()
    ssim = SsimMetric()
    lpips_metric = LpipsMetric(device=device)
    niqe_metric = NiqeMetric(device=device)
    noise_config = load_noise_config(ROOT / "configs" / "data" / "noise.yaml")
    images = {row["low_path"]: load_rgb(ROOT / row["low_path"]) for row in rows}
    targets = {row["low_path"]: load_rgb(ROOT / row["gt_path"]) for row in rows}
    darken_gain = 0.2

    for group in args.groups:
        spec = GROUP_SPECS[group]
        out_dir = args.out_root / group
        out_dir.mkdir(parents=True, exist_ok=True)
        lock_path = out_dir / "test_lock.json"
        if is_test and lock_path.exists():
            raise SystemExit(f"refusing to rerun locked test: {lock_path}")
        result_rows: list[dict[str, object]] = []
        checkpoint_hashes: dict[str, str] = {}
        for arm, (root_text, mode) in spec["arms"].items():  # type: ignore[union-attr]
            root = ROOT / str(root_text)
            for seed in args.seeds:
                checkpoint_path = root / f"seed{seed}" / "best.pt"
                if not checkpoint_path.exists():
                    raise SystemExit(f"missing checkpoint: {checkpoint_path}")
                checkpoint_hashes[str(checkpoint_path.relative_to(ROOT))] = sha256(checkpoint_path)
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                model = build_model(checkpoint["config"]["model"])
                model.load_state_dict(checkpoint["model_state"])
                model.to(device).eval()
                for level in spec["levels"]:  # type: ignore[union-attr]
                    print(f"group={group} arm={arm} seed={seed} level={level}")
                    group_rows: list[dict[str, object]] = []
                    predictions: list[np.ndarray] = []
                    group_targets: list[np.ndarray] = []
                    for row in rows:
                        image_id = row["low_path"]
                        target = targets[image_id]
                        source, expected = source_and_target(
                            row,
                            image=images[image_id],
                            target=target,
                            mode=str(mode),
                            level=str(level),
                            darken_gain=darken_gain,
                            noise_config=noise_config,
                        )
                        tensor = torch.from_numpy(np.ascontiguousarray(source.astype(np.float32)))
                        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device)
                        started = time.perf_counter()
                        with torch.inference_mode():
                            prediction = denoise_full_image(model, tensor)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        inference_ms = (time.perf_counter() - started) * 1000.0
                        prediction_np = prediction.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        scores = metric_values(
                            {"psnr": psnr, "ssim": ssim}, prediction_np, expected
                        )
                        group_rows.append(
                            {
                                "run_id": "stage07-ablation-evaluation",
                                "group": group,
                                "arm": arm,
                                "dataset": row["dataset"],
                                "split": row["split"],
                                "seed": seed,
                                "noise_level": level,
                                "image_id": image_id,
                                "psnr": scores["psnr"],
                                "ssim": scores["ssim"],
                                "lpips": 0.0,
                                "niqe": 0.0,
                                "inference_ms": inference_ms,
                                "width": expected.shape[1],
                                "height": expected.shape[0],
                            }
                        )
                        predictions.append(prediction_np)
                        group_targets.append(expected)
                    lpips_scores = batch_lpips(
                        lpips_metric, predictions, group_targets, batch=args.batch_size
                    )
                    niqe_scores = batch_niqe(niqe_metric, predictions, batch=args.batch_size)
                    for row, lpips_score, niqe_score in zip(
                        group_rows, lpips_scores, niqe_scores, strict=True
                    ):
                        row["lpips"] = lpips_score
                        row["niqe"] = niqe_score
                    result_rows.extend(group_rows)
                    print(
                        f"  {arm} s{seed} {level}: "
                        f"PSNR={np.mean([float(r['psnr']) for r in group_rows]):.3f}"
                    )
        per_image_path = out_dir / "per_image.csv"
        write_csv_rows(
            per_image_path,
            [{key: format_cell(value) for key, value in row.items()} for row in result_rows],
            PER_IMAGE_COLUMNS,
        )
        metadata = {
            "run_id": "stage07-ablation-evaluation",
            "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
            "python": platform.python_version(),
            "device": str(device),
            "torch": torch.__version__,
            "test_used": is_test,
            "group": group,
            "splits": list(args.splits),
            "limit": args.limit,
            "n_images": len(rows),
            "n_rows": len(result_rows),
            "seeds": args.seeds,
            "levels": list(spec["levels"]),  # type: ignore[arg-type]
            "darken_gain": darken_gain if group.startswith("synthetic") else None,
            "checkpoints": checkpoint_hashes,
            "split_hashes": split_manifest.get("files_sha256", {}),
        }
        (out_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        outputs = {"per_image.csv": sha256(per_image_path)}
        if is_test:
            lock = {
                "used_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "group": group,
                "split_hashes": split_manifest.get("files_sha256", {}),
                "checkpoints": checkpoint_hashes,
                "outputs": outputs,
            }
            lock_path.write_text(
                json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        print(f"group={group} rows={len(result_rows)} out={per_image_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
