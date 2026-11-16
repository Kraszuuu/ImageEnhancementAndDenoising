"""Run metadata helpers shared by training and evaluation scripts."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

PACKAGES = [
    "numpy",
    "scipy",
    "scikit-image",
    "lpips",
    "pyiqa",
    "torch",
    "torchvision",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def git_state(root: Path) -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def environment(device: str | torch.device) -> dict[str, object]:
    resolved = torch.device(device)
    payload: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": package_versions(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(resolved),
        "cuda_available": torch.cuda.is_available(),
    }
    if resolved.type == "cuda" and torch.cuda.is_available():
        payload["gpu"] = torch.cuda.get_device_name(resolved)
        payload["vram_total_mib"] = round(
            torch.cuda.get_device_properties(resolved).total_memory / 2**20, 1
        )
    return payload


def write_json(path: Path, payload: dict[str, object]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
