"""Reproduce data, training, evaluation and final analysis for the project.

The default run reuses existing checkpoints and locked test outputs. Pass
``--force`` to rerun training and non-locked evaluations. Locked test outputs
are never overwritten by this script.

Examples:
    python scripts/reproduceAll.py --stage all --dry-run
    python scripts/reproduceAll.py --stage data --allow-download
    python scripts/reproduceAll.py --stage analysis
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "results" / "final"
PACKAGES = (
    "numpy",
    "scipy",
    "scikit-image",
    "opencv-python-headless",
    "Pillow",
    "matplotlib",
    "pandas",
    "torch",
    "torchvision",
    "lpips",
    "bm3d",
    "bm4d",
    "pyiqa",
    "streamlit",
    "pytest",
    "ruff",
)
TOLERANCE = {"psnr_db": 0.1, "ssim": 0.001}
STAGES = ("data", "metrics", "denoising", "brightening", "combined", "analysis")


@dataclass(frozen=True)
class Step:
    name: str
    script: str
    args: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    locked: bool = False
    always: bool = False
    force_flag: bool = False
    requires_download: bool = False

    def command(self, python: str, *, force: bool = False) -> list[str]:
        payload = [python, str(ROOT / "scripts" / self.script), *self.args]
        if force and self.force_flag:
            payload.append("--force")
        return payload


def _seed_args(seeds: list[int]) -> tuple[str, ...]:
    return tuple(str(seed) for seed in seeds)


def build_stage_steps(seeds: list[int]) -> dict[str, list[Step]]:
    seed_values = _seed_args(seeds)
    data = [
        Step(
            "download LOL",
            "downloadData.py",
            ("--dataset", "lol"),
            outputs=("data/raw/lol",),
            requires_download=True,
        ),
        Step(
            "download LOL-v2 Real",
            "downloadData.py",
            ("--dataset", "lolv2-real"),
            outputs=("data/raw/lolv2-real",),
            requires_download=True,
        ),
        Step(
            "record LOL-v2 Synthetic backup metadata",
            "downloadData.py",
            ("--dataset", "lolv2-synthetic", "--backup-only"),
            requires_download=True,
        ),
        Step("audit datasets", "auditData.py", always=True),
        Step("build manifests", "buildManifests.py", always=True),
        Step("build frozen splits", "buildSplits.py", always=True),
    ]
    metrics = [
        Step("validate metric fixtures", "validateMetrics.py", always=True),
        Step("calibrate controlled noise", "calibrateNoise.py", always=True),
    ]
    denoising = [
        Step(
            "tune classical denoisers",
            "tuneClassicalDenoising.py",
            outputs=("configs/classical/denoising_frozen.yaml",),
        ),
        Step(
            "evaluate classical denoising",
            "evaluateClassicalDenoising.py",
            outputs=("results/classical_denoising/per_image.csv",),
            locked=True,
            force_flag=True,
        ),
        Step("classical denoising figures", "makeClassicalFigures.py", always=True),
        Step(
            "benchmark U-Net",
            "benchmarkUnet.py",
            outputs=("results/denoising_cnn/benchmark.csv",),
            always=True,
        ),
        *[
            Step(
                f"train denoising U-Net seed {seed}",
                "trainDenoising.py",
                (
                    "--config",
                    "configs/training/denoising.yaml",
                    "--seed",
                    str(seed),
                ),
                outputs=(f"checkpoints/denoising_cnn/seed{seed}/best.pt",),
            )
            for seed in seeds
        ],
        Step(
            "evaluate denoising U-Net",
            "evaluateDenoising.py",
            outputs=("results/denoising_cnn/evaluation/per_image.csv",),
            locked=True,
            force_flag=True,
        ),
        Step(
            "evaluate D_fixed reference",
            "evaluateDFixed.py",
            outputs=("results/denoising_cnn/evaluation_d_fixed/per_image.csv",),
            locked=True,
            force_flag=True,
        ),
        Step("benchmark U-Net timing", "benchmarkCnnTiming.py", always=True),
        Step("compare denoising methods", "compareDenoising.py", always=True),
        Step("denoising figures", "makeDenoisingFigures.py", always=True),
    ]
    brightening = [
        Step(
            "tune classical brighteners",
            "tuneClassicalBrightening.py",
            outputs=("configs/classical/brightening_frozen.yaml",),
        ),
        Step(
            "evaluate classical brightening",
            "evaluateClassicalBrightening.py",
            outputs=("results/classical_brightening/per_image.csv",),
            locked=True,
            force_flag=True,
        ),
        Step(
            "precompute D_fixed sources",
            "precomputeDFixed.py",
            outputs=("outputs/d_fixed_cache/cache_manifest.json",),
            force_flag=True,
        ),
        *[
            Step(
                f"train raw-track brightener seed {seed}",
                "trainBrightening.py",
                ("--track", "raw", "--seed", str(seed)),
                outputs=(f"checkpoints/brightening_cnn/raw/seed{seed}/best.pt",),
            )
            for seed in seeds
        ],
        *[
            Step(
                f"train D_fixed-track brightener seed {seed}",
                "trainBrightening.py",
                ("--track", "d_fixed", "--seed", str(seed)),
                outputs=(f"checkpoints/brightening_cnn/d_fixed/seed{seed}/best.pt",),
            )
            for seed in seeds
        ],
        Step(
            "evaluate CNN brighteners",
            "evaluateBrightening.py",
            outputs=("results/brightening_cnn/evaluation/per_image.csv",),
            locked=True,
            force_flag=True,
        ),
    ]
    combined = [
        *[
            Step(
                f"precompute CNN denoiser sources seed {seed}",
                "precomputePipelineSources.py",
                (
                    "--checkpoint",
                    f"checkpoints/denoising_cnn/seed{seed}/best.pt",
                    "--cache-dir",
                    f"outputs/pipeline_source_cache/cnn/seed{seed}",
                ),
                outputs=(
                    f"outputs/pipeline_source_cache/cnn/seed{seed}/cache_manifest.json",
                ),
                force_flag=True,
            )
            for seed in seeds
        ],
        *[
            Step(
                f"train pipeline brightener {denoiser} seed {seed}",
                "trainPipelineBrightener.py",
                ("--denoiser", denoiser, "--seed", str(seed)),
                outputs=(
                    f"checkpoints/pipeline_brighteners/{denoiser}/seed{seed}/best.pt",
                ),
            )
            for denoiser in ("classical", "cnn")
            for seed in seeds
        ],
        *[
            Step(
                f"train end-to-end {mode} seed {seed}",
                "trainEndToEnd.py",
                ("--input-mode", mode, "--seed", str(seed)),
                outputs=(f"checkpoints/end_to_end/{mode}/seed{seed}/best.pt",),
            )
            for mode in ("real", "controlled")
            for seed in seeds
        ],
        Step(
            "evaluate combined pipelines",
            "evaluatePipelines.py",
            outputs=("results/pipelines/evaluation/per_image.csv",),
            locked=True,
            force_flag=True,
        ),
        Step("pipeline figures", "makePipelineFigures.py", always=True),
    ]
    analysis = [
        *[
            Step(
                f"Stage 07 L1 loss seed {seed}",
                "trainEndToEnd.py",
                (
                    "--input-mode",
                    "real",
                    "--seed",
                    str(seed),
                    "--loss-name",
                    "l1",
                    "--tag",
                    f"stage07-pilot-l1-seed{seed}",
                    "--out-dir",
                    f"results/ablations/pilot/loss_l1/real/seed{seed}",
                    "--checkpoint-dir",
                    f"checkpoints/ablations/pilot/loss_l1/real/seed{seed}",
                ),
                outputs=(
                    f"checkpoints/ablations/pilot/loss_l1/real/seed{seed}/best.pt",
                ),
            )
            for seed in seeds
        ],
        *[
            Step(
                f"Stage 07 composition {arm} seed {seed}",
                "trainEndToEnd.py",
                (
                    "--input-mode",
                    "real",
                    "--seed",
                    str(seed),
                    "--tag",
                    f"stage07-composition-{arm}-seed{seed}",
                    "--out-dir",
                    f"results/ablations/composition/{arm}/real/seed{seed}",
                    "--checkpoint-dir",
                    f"checkpoints/ablations/composition/{arm}/real/seed{seed}",
                    "--train-splits",
                    f"data/splits/{'lol' if arm == 'lol_only' else 'lolv2'}_train.csv",
                ),
                outputs=(
                    f"checkpoints/ablations/composition/{arm}/real/seed{seed}/best.pt",
                ),
            )
            for arm in ("lol_only", "lolv2_only")
            for seed in seeds
        ],
        *[
            Step(
                f"Stage 07 synthetic {mode} seed {seed}",
                "trainEndToEnd.py",
                (
                    "--input-mode",
                    mode,
                    "--seed",
                    str(seed),
                    "--tag",
                    f"stage07-{mode}-seed{seed}",
                    "--out-dir",
                    f"results/ablations/synthetic/{mode}/seed{seed}",
                    "--checkpoint-dir",
                    f"checkpoints/ablations/synthetic/{mode}/seed{seed}",
                ),
                outputs=(
                    f"checkpoints/ablations/synthetic/{mode}/seed{seed}/best.pt",
                ),
            )
            for mode in ("synthetic_brightening", "synthetic_denoising")
            for seed in seeds
        ],
        Step(
            "evaluate Stage 07 ablations",
            "evaluateAblations.py",
            (
                "--groups",
                "loss_l1",
                "composition",
                "synthetic_brightening",
                "synthetic_denoising",
                "--seeds",
                *seed_values,
            ),
            outputs=(
                "results/ablations/evaluation/loss_l1/test_lock.json",
                "results/ablations/evaluation/composition/test_lock.json",
                "results/ablations/evaluation/synthetic_brightening/test_lock.json",
                "results/ablations/evaluation/synthetic_denoising/test_lock.json",
            ),
            locked=True,
        ),
        Step("aggregate unified results", "analyzeResults.py", always=True),
        Step("make analysis figures", "makeFigures.py", always=True),
        Step("make failure panels", "makeFailurePanels.py", always=True),
    ]
    return {
        "data": data,
        "metrics": metrics,
        "denoising": denoising,
        "brightening": brightening,
        "combined": combined,
        "analysis": analysis,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_info() -> dict[str, object]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
    }


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def cuda_info() -> dict[str, object]:
    info: dict[str, object] = {"available": None, "driver_version": None}
    try:
        import torch
    except ImportError:
        return info
    info["torch"] = torch.__version__
    info["cuda_version"] = torch.version.cuda
    info["available"] = torch.cuda.is_available()
    try:
        info["device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            info["driver_version"] = result.stdout.strip().splitlines()[0]
    except (OSError, IndexError):
        pass
    return info


def file_hashes(paths: list[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in paths:
        if path.is_file():
            result[path.relative_to(ROOT).as_posix()] = sha256(path)
    return result


def config_and_split_hashes() -> dict[str, str]:
    paths = [
        *sorted((ROOT / "configs").rglob("*.yaml")),
        *sorted((ROOT / "data" / "splits").glob("*.csv")),
        *sorted((ROOT / "data" / "splits").glob("*.json")),
        *sorted((ROOT / "data" / "manifests").glob("*.csv")),
        *sorted((ROOT / "data" / "manifests").glob("*.json")),
    ]
    return file_hashes(paths)


def base_metadata(args: argparse.Namespace, stages: list[str], python: str) -> dict[str, object]:
    lock = ROOT / "requirements.lock.txt"
    return {
        "script": "scripts/reproduceAll.py",
        "started_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "running",
        "stage": args.stage,
        "stages": stages,
        "seeds": args.seeds,
        "force": args.force,
        "allow_download": args.allow_download,
        "skip_tests": args.skip_tests,
        "python": {
            "executable": python,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": platform.platform(),
        "git": git_info(),
        "packages": package_versions(),
        "cuda": cuda_info(),
        "requirements_lock_sha256": sha256(lock) if lock.exists() else None,
        "config_and_split_sha256": config_and_split_hashes(),
        "reproduction_tolerance": TOLERANCE,
        "notes": [
            "Locked test outputs are not rerun by reproduceAll.py.",
            "Raw data and checkpoints are intentionally ignored by git.",
            "Bitwise-identical retraining is not guaranteed.",
        ],
        "steps": [],
    }


def write_metadata(metadata: dict[str, object]) -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    (FINAL_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def command_line(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def execute(
    command: list[str],
    *,
    env: dict[str, str],
    dry_run: bool,
) -> int:
    print(command_line(command), flush=True)
    if dry_run:
        return 0
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


def step_exists(step: Step) -> bool:
    return bool(step.outputs) and all((ROOT / item).exists() for item in step.outputs)


def run_step(
    step: Step,
    *,
    python: str,
    env: dict[str, str],
    args: argparse.Namespace,
    metadata: dict[str, object],
) -> int:
    if (
        step_exists(step)
        and not step.always
        and not (args.force and not step.locked)
    ):
        reason = "locked test output exists" if step.locked else "output exists"
        print(f"SKIP {step.name}: {reason}")
        if args.dry_run:
            command = step.command(python, force=args.force)
            print(f"  {command_line(command)}  # would be skipped")
        metadata["steps"].append(  # type: ignore[union-attr]
            {"name": step.name, "status": "skipped", "reason": reason}
        )
        return 0
    command = step.command(python, force=args.force)
    if step.requires_download and not args.allow_download and not args.dry_run:
        print(
            f"SKIP {step.name}: pass --allow-download to permit network downloads",
            file=sys.stderr,
        )
        return 2
    print(f"RUN {step.name}", flush=True)
    code = execute(command, env=env, dry_run=args.dry_run)
    metadata["steps"].append(  # type: ignore[union-attr]
        {
            "name": step.name,
            "status": "dry-run" if args.dry_run else ("completed" if code == 0 else "failed"),
            "command": command_line(command),
            "returncode": code,
        }
    )
    return code


def run_data(
    steps: list[Step],
    *,
    python: str,
    env: dict[str, str],
    args: argparse.Namespace,
    metadata: dict[str, object],
) -> int:
    check = [python, str(ROOT / "scripts" / "downloadData.py"), "--check"]
    download_steps = [step for step in steps if step.requires_download]
    rebuild_steps = [step for step in steps if not step.requires_download]
    if args.dry_run:
        print(command_line(check))
        for step in download_steps:
            suffix = "" if args.allow_download else "  # needs --allow-download"
            print(command_line(step.command(python, force=args.force)) + suffix)
        for step in rebuild_steps:
            code = run_step(
                step, python=python, env=env, args=args, metadata=metadata
            )
            if code != 0:
                return code
        return 0

    verified = subprocess.run(check, cwd=ROOT, env=env, check=False).returncode == 0
    if not verified:
        if not args.allow_download:
            print(
                "Data hashes are missing or invalid. Re-run with --allow-download.",
                file=sys.stderr,
            )
            return 2
        for step in download_steps:
            code = run_step(
                step, python=python, env=env, args=args, metadata=metadata
            )
            if code != 0:
                return code
        verified = subprocess.run(check, cwd=ROOT, env=env, check=False).returncode == 0
        if not verified:
            print("Dataset hash verification still fails after download.", file=sys.stderr)
            return 2
    else:
        print("Data hashes verified; download steps are skipped.")
        metadata["steps"].append(  # type: ignore[union-attr]
            {"name": "verify dataset hashes", "status": "completed"}
        )
    for step in rebuild_steps:
        code = run_step(step, python=python, env=env, args=args, metadata=metadata)
        if code != 0:
            return code
    return 0


def package_final_outputs(*, dry_run: bool) -> None:
    print("PACKAGE final tables and figures into results/final")
    if dry_run:
        return
    tables_dir = FINAL_DIR / "tables"
    figures_dir = FINAL_DIR / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted((ROOT / "results" / "ablations").glob("*.csv")):
        shutil.copy2(path, tables_dir / path.name)
    source_figures = ROOT / "results" / "ablations" / "figures"
    if source_figures.exists():
        for path in sorted(source_figures.glob("*.png")):
            shutil.copy2(path, figures_dir / path.name)
    lines = [
        "# Final results package",
        "",
        "Generated from committed evaluation outputs and figures.",
        "",
        f"Generated at: {datetime.now(UTC).isoformat(timespec='seconds')}",
        "",
        "## Tables",
        "",
        *[
            f"- `tables/{path.name}`"
            for path in sorted(tables_dir.glob("*.csv"))
        ],
        "",
        "## Figures",
        "",
        *[
            f"- `figures/{path.name}`"
            for path in sorted(figures_dir.glob("*.png"))
        ],
        "",
        "Failure panels are large source PNGs under "
        "`results/ablations/failure_panels/` and are linked from the final handout.",
        "",
        "See `Outdated/Plans/handouts/final/HANDOUT.md` for interpretation.",
        "",
    ]
    (FINAL_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", *STAGES), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands and validate the plan without side effects",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun training and non-locked steps even if outputs exist",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="permit pinned dataset downloads when hashes are missing",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="skip the pytest preflight",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for child commands (default: current)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < len(args.seeds) <= 3:
        raise SystemExit("--seeds accepts one to three seed values")
    stages = list(STAGES) if args.stage == "all" else [args.stage]
    plans = build_stage_steps(args.seeds)
    metadata = base_metadata(args, stages, args.python)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    started = time.perf_counter()
    exit_code = 0

    if not args.dry_run:
        write_metadata(metadata)

    try:
        if not args.skip_tests:
            test_command = [args.python, "-m", "pytest", "-q"]
            print("RUN pytest preflight", flush=True)
            code = execute(test_command, env=env, dry_run=args.dry_run)
            metadata["steps"].append(  # type: ignore[union-attr]
                {
                    "name": "pytest preflight",
                    "status": (
                        "dry-run"
                        if args.dry_run
                        else ("completed" if code == 0 else "failed")
                    ),
                    "command": command_line(test_command),
                    "returncode": code,
                }
            )
            if code != 0:
                exit_code = code
                return code

        for stage in stages:
            print(f"\n=== stage: {stage} ===", flush=True)
            if stage == "data":
                code = run_data(
                    plans[stage],
                    python=args.python,
                    env=env,
                    args=args,
                    metadata=metadata,
                )
            else:
                code = 0
                for step in plans[stage]:
                    code = run_step(
                        step,
                        python=args.python,
                        env=env,
                        args=args,
                        metadata=metadata,
                    )
                    if code != 0:
                        break
            if code != 0:
                exit_code = code
                break

        if exit_code == 0 and "analysis" in stages:
            package_final_outputs(dry_run=args.dry_run)

        return exit_code
    finally:
        metadata["finished_at_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
        metadata["duration_seconds"] = round(time.perf_counter() - started, 3)
        metadata["status"] = "dry-run" if args.dry_run else (
            "completed" if exit_code == 0 else "failed"
        )
        metadata["exit_code"] = exit_code
        if not args.dry_run:
            write_metadata(metadata)


if __name__ == "__main__":
    raise SystemExit(main())

