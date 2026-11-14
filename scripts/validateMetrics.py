"""Validate Stage 01 metrics against references and write fixtures."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lowlight.metrics import (  # noqa: E402
    ImagePair,
    InferenceTimer,
    LpipsMetric,
    MetricsRunner,
    NiqeMetric,
    PsnrMetric,
    SsimMetric,
    resolve_device,
    to_luminance,
)

TOLERANCES = {
    "psnr_db": 1e-6,
    "ssim": 1e-6,
    "lpips": 1e-6,
    "niqe": 1e-4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "results" / "metrics"
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def synthetic_pair(seed: int = 0, size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic textured target and a degraded prediction."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    channels = []
    for phase in (0.0, 0.8, 1.7):
        wave = np.sin(xx / 7.0 + phase) * np.cos(yy / 11.0 - phase)
        channels.append(0.45 + 0.22 * wave)
    target = np.stack(channels, axis=2)
    target = np.clip(target + 0.02 * rng.standard_normal(target.shape), 0.0, 1.0)
    prediction = 0.88 * target + 0.06 + 0.01 * rng.standard_normal(target.shape)
    return np.clip(prediction, 0.0, 1.0), target


def _check(
    name: str,
    value: float,
    reference: float,
    tolerance: float,
) -> dict[str, object]:
    difference = abs(value - reference)
    return {
        "name": name,
        "value": value,
        "reference": reference,
        "abs_diff": difference,
        "tolerance": tolerance,
        "passed": bool(difference <= tolerance),
    }


def _timing(device_name: str) -> dict[str, object]:
    import torch

    torch.manual_seed(0)
    model = torch.nn.Conv2d(3, 3, 3, padding=1)
    model = model.to(device_name).eval()
    tensor = torch.rand(1, 3, 64, 64, device=device_name)
    forward = lambda: model(tensor)  # noqa: E731
    with torch.inference_mode():
        result = InferenceTimer(warmup=10, iterations=50).measure(
            forward, device=device_name
        )
    return result.as_dict()


def main() -> int:
    args = parse_args()
    device = resolve_device(None if args.device == "auto" else args.device)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    prediction, target = synthetic_pair()
    identical = target.copy()

    psnr_metric = PsnrMetric()
    psnr_luma_metric = PsnrMetric(channel="luma")
    ssim_metric = SsimMetric()
    lpips_metric = LpipsMetric(device=device)
    niqe_metric = NiqeMetric(device=device)

    checks = []
    psnr_reference = 10.0 * math.log10(1.0 / float(np.mean((prediction - target) ** 2)))
    checks.append(
        _check(
            "psnr_rgb",
            psnr_metric.compute(prediction, target),
            psnr_reference,
            TOLERANCES["psnr_db"],
        )
    )
    luma_reference = 10.0 * math.log10(
        1.0
        / float(np.mean((to_luminance(prediction) - to_luminance(target)) ** 2))
    )
    checks.append(
        _check(
            "psnr_luma",
            psnr_luma_metric.compute(prediction, target),
            luma_reference,
            TOLERANCES["psnr_db"],
        )
    )

    ssim_reference = float(
        structural_similarity(
            target,
            prediction,
            channel_axis=2,
            data_range=1.0,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    )
    checks.append(
        _check("ssim", ssim_metric.compute(prediction, target), ssim_reference, TOLERANCES["ssim"])
    )

    lpips_zero = lpips_metric.compute(identical, identical)
    checks.append(_check("lpips_identical", lpips_zero, 0.0, TOLERANCES["lpips"]))
    lpips_first = lpips_metric.compute(prediction, target)
    lpips_second = lpips_metric.compute(prediction, target)
    checks.append(
        _check("lpips_deterministic", lpips_first, lpips_second, TOLERANCES["lpips"])
    )

    niqe_first = niqe_metric.compute(prediction, None)
    niqe_second = niqe_metric.compute(prediction, None)
    checks.append(
        _check("niqe_deterministic", niqe_first, niqe_second, TOLERANCES["niqe"])
    )
    niqe_finite = {
        "name": "niqe_finite",
        "value": niqe_first,
        "reference": None,
        "abs_diff": None,
        "tolerance": None,
        "passed": bool(math.isfinite(niqe_first) and niqe_first > 0.0),
    }
    checks.append(niqe_finite)

    timing = {"cpu": _timing("cpu")}
    import torch

    if torch.cuda.is_available():
        timing["cuda"] = _timing("cuda")
    else:
        timing["cuda"] = {"available": False}

    runner = MetricsRunner(
        [psnr_metric, ssim_metric, lpips_metric, niqe_metric],
        run_id="stage01-fixture",
        method="affine",
        dataset="fixture",
        split="val",
        noise_level="clean",
        seed=0,
        split_hash="fixture",
        bootstrap_resamples=1000,
    )
    pairs = []
    base_prediction, base_target = synthetic_pair(seed=1)
    for index, gain in enumerate((0.75, 0.85, 0.95, 1.05)):
        pairs.append(
            ImagePair(
                image_id=f"fixture{index}",
                prediction=np.clip(gain * base_prediction, 0.0, 1.0),
                target=base_target,
            )
        )
    fixture_dir = out_dir / "fixtures"
    rows = runner.run(pairs, fixture_dir)

    versions = {
        package: version(package)
        for package in ("numpy", "scipy", "scikit-image", "lpips", "pyiqa", "torch")
    }
    weights = {
        "lpips": lpips_metric.metadata().get("weights_sha256"),
        "lpips_backbone": lpips_metric.metadata().get("backbone_weights_sha256"),
        "niqe_params": niqe_metric.metadata().get("params_sha256"),
    }
    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "device": str(device),
        "versions": versions,
        "weights_sha256": weights,
        "tolerances": TOLERANCES,
        "checks": checks,
        "timing": timing,
        "fixtures": {
            "rows": len(rows),
            "per_image": str((fixture_dir / "per_image.csv").relative_to(ROOT)),
            "summary": str((fixture_dir / "summary.csv").relative_to(ROOT)),
        },
    }
    (out_dir / "validation.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "metric_versions.json").write_text(
        json.dumps(
            {
                "created_at_utc": payload["created_at_utc"],
                "versions": versions,
                "weights_sha256": weights,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"python={platform.python_version()} device={device}")
    print(f"{'check':<20} {'value':>12} {'reference':>12} {'abs_diff':>12}  result")
    for item in checks:
        value = "-" if item["value"] is None else f"{float(item['value']):.6f}"
        reference = (
            "-" if item["reference"] is None else f"{float(item['reference']):.6f}"
        )
        difference = (
            "-" if item["abs_diff"] is None else f"{float(item['abs_diff']):.3e}"
        )
        status = "PASS" if item["passed"] else "FAIL"
        print(f"{item['name']:<20} {value:>12} {reference:>12} {difference:>12}  {status}")
    for name, result in timing.items():
        if "mean_ms" in result:
            print(
                f"timing[{name}] mean={result['mean_ms']:.3f} ms "
                f"std={result['std_ms']:.3f} median={result['median_ms']:.3f}"
            )
    print(f"lpips(diff)={lpips_first:.6f} niqe={niqe_first:.6f}")
    print(f"report={out_dir / 'validation.json'}")
    print(f"fixtures={fixture_dir}")

    return 0 if all(item["passed"] for item in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
