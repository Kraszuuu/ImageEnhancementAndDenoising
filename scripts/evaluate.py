"""Dispatch frozen evaluation scripts for one project stage.

Examples:
    python scripts/evaluate.py classical-denoising
    python scripts/evaluate.py pipelines --seeds 0
    python scripts/evaluate.py ablations --groups loss_l1
    python scripts/evaluate.py all --dry-run

Arguments after the stage are forwarded to the selected evaluation script.
Locked test outputs are protected by the underlying scripts.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EVALUATORS = {
    "classical-denoising": "evaluateClassicalDenoising.py",
    "cnn-denoising": "evaluateDenoising.py",
    "d-fixed": "evaluateDFixed.py",
    "classical-brightening": "evaluateClassicalBrightening.py",
    "cnn-brightening": "evaluateBrightening.py",
    "pipelines": "evaluatePipelines.py",
    "ablations": "evaluateAblations.py",
}

ALL_ORDER = (
    "classical-denoising",
    "cnn-denoising",
    "d-fixed",
    "classical-brightening",
    "cnn-brightening",
    "pipelines",
    "ablations",
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(*EVALUATORS, "all"),
        help="evaluation stage to run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands without running them",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for child scripts (default: current)",
    )
    args, extra = parser.parse_known_args()
    if extra[:1] == ["--"]:
        extra = extra[1:]
    return args, extra


def build_commands(stage: str, python: str, extra: list[str]) -> list[list[str]]:
    selected = ALL_ORDER if stage == "all" else (stage,)
    if stage == "all" and extra:
        raise SystemExit("extra arguments are not supported with --stage all")
    return [
        [python, str(ROOT / "scripts" / EVALUATORS[item]), *extra]
        for item in selected
    ]


def main() -> int:
    args, extra = parse_args()
    commands = build_commands(args.stage, args.python, extra)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    for command in commands:
        printable = subprocess.list2cmdline(command)
        if args.dry_run:
            print(printable)
            continue
        print(printable, flush=True)
        result = subprocess.run(command, cwd=ROOT, env=env, check=False)
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
