from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_reproduce_all_dry_run_passes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "reproduceAll.py"),
            "--stage",
            "all",
            "--dry-run",
            "--skip-tests",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "downloadData.py --check" in result.stdout
    assert "evaluateAblations.py" in result.stdout
    assert "PACKAGE final tables and figures" in result.stdout


def test_evaluate_dry_run_passes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate.py"),
            "--stage",
            "pipelines",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "evaluatePipelines.py" in result.stdout

def test_force_reruns_nonlocked_but_keeps_locked_outputs() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "reproduceAll.py"),
            "--stage",
            "denoising",
            "--dry-run",
            "--skip-tests",
            "--force",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "RUN tune classical denoisers" in result.stdout
    assert "SKIP evaluate classical denoising: locked test output exists" in result.stdout
