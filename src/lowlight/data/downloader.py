"""Pinned Hugging Face dataset downloader.

Raw datasets are never committed. Every download is pinned to a 40-character
commit hash and hashed with SHA256; the resulting metadata is written to
``data/manifests/dataset_versions.json``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

LOGGER = logging.getLogger(__name__)

COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class DatasetSpec:
    """Static description of a supported dataset."""

    name: str
    repo_id: str
    revision: str
    license: str
    status: str
    archive: str | None = None
    allow_patterns: tuple[str, ...] = ()
    description: str = ""


DATASETS: dict[str, DatasetSpec] = {
    "lol": DatasetSpec(
        name="lol",
        repo_id="geekyrakshit/LoL-Dataset",
        revision="bdef22d0f5803da8775114496a6689333265e157",
        license="unknown",
        status="primary",
        archive="lol_dataset.zip",
        description="LOL: 485 train pairs (our485) and 15 test pairs (eval15).",
    ),
    "lolv2-real": DatasetSpec(
        name="lolv2-real",
        repo_id="okhater/lolv2-real",
        revision="db2979107f7c22d372c6e4f1b06fd6203069af0a",
        license="apache-2.0",
        status="primary",
        archive=None,
        allow_patterns=("Train/**", "Test/**", "LICENSE", "README.md"),
        description="LOL-v2 Real: 689 train pairs and 100 test pairs.",
    ),
    "lolv2-synthetic": DatasetSpec(
        name="lolv2-synthetic",
        repo_id="okhater/lolv2-synthetic",
        revision="2fe16f9ebde66b20df7222999f53843f68c61aa4",
        license="apache-2.0",
        status="backup",
        description="LOL-v2 Synthetic: backup only, never downloaded by default.",
    ),
}


@dataclass
class DownloadResult:
    dataset: str
    repo_id: str
    revision: str
    license: str
    status: str
    path: str
    sha256: str | None
    size_bytes: int
    downloaded_at_utc: str
    duration_seconds: float
    extra: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_revision(revision: str) -> str:
    if not COMMIT_RE.match(revision):
        raise ValueError(
            f"revision must be a 40-character commit hash, got {revision!r}"
        )
    return revision


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class HfDatasetDownloader:
    """Downloads a single pinned dataset revision into ``data/raw``."""

    def __init__(self, spec: DatasetSpec) -> None:
        self.spec = spec
        validate_revision(spec.revision)

    def download_file(self, filename: str, dest_dir: Path, force: bool = False) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / filename
        if target.exists() and not force:
            LOGGER.info("exists, skipping download: %s", target)
            return target
        LOGGER.info(
            "downloading %s/%s@%s",
            self.spec.repo_id,
            filename,
            self.spec.revision[:12],
        )
        cached = hf_hub_download(
            repo_id=self.spec.repo_id,
            filename=filename,
            revision=self.spec.revision,
            repo_type="dataset",
        )
        shutil.copy2(cached, target)
        return target

    def download_snapshot(
        self, dest_dir: Path, allow_patterns: tuple[str, ...], force: bool = False
    ) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        if any(dest_dir.iterdir()) and not force:
            LOGGER.info("snapshot directory non-empty, skipping: %s", dest_dir)
            return dest_dir
        LOGGER.info(
            "snapshot %s@%s -> %s",
            self.spec.repo_id,
            self.spec.revision[:12],
            dest_dir,
        )
        snapshot_download(
            repo_id=self.spec.repo_id,
            revision=self.spec.revision,
            repo_type="dataset",
            local_dir=dest_dir,
            allow_patterns=list(allow_patterns),
        )
        return dest_dir

    def fetcher(self, raw_root: Path, force: bool = False) -> DownloadResult:
        """Download the dataset according to its spec and return metadata."""
        started = time.perf_counter()
        dataset_root = raw_root / self.spec.name
        dataset_root.mkdir(parents=True, exist_ok=True)

        if self.spec.archive:
            archive_path = self.download_file(self.spec.archive, dataset_root, force)
            digest = sha256_file(archive_path)
            size = archive_path.stat().st_size
            extracted = dataset_root / "extracted"
            if force or not extracted.exists():
                if extracted.exists():
                    shutil.rmtree(extracted)
                with zipfile.ZipFile(archive_path) as zf:
                    zf.extractall(extracted)
                self._remove_ds_store(extracted)
            path = extracted
            extra = {"archive": self.spec.archive, "archive_sha256": digest}
        else:
            if not self.spec.allow_patterns:
                raise ValueError(f"{self.spec.name} has no archive or allow_patterns")
            path = self.download_snapshot(dataset_root, self.spec.allow_patterns, force)
            digest = None
            size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
            extra = {"allow_patterns": list(self.spec.allow_patterns)}

        duration = time.perf_counter() - started
        result = DownloadResult(
            dataset=self.spec.name,
            repo_id=self.spec.repo_id,
            revision=self.spec.revision,
            license=self.spec.license,
            status=self.spec.status,
            path=str(path.as_posix()),
            sha256=digest,
            size_bytes=size,
            downloaded_at_utc=utc_now(),
            duration_seconds=round(duration, 3),
            extra=extra,
        )
        LOGGER.info(
            "downloaded %s: %d bytes in %.1fs",
            self.spec.name,
            size,
            duration,
        )
        return result

    @staticmethod
    def _remove_ds_store(root: Path) -> None:
        for junk in root.rglob(".DS_Store"):
            junk.unlink()


def update_dataset_versions(
    manifest_path: Path, result: DownloadResult
) -> dict[str, object]:
    """Insert or replace one dataset entry in ``dataset_versions.json``."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        data = {}
    data[result.dataset] = result.as_dict()
    manifest_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return data


def record_backup_only(
    spec: DatasetSpec, manifest_path: Path
) -> dict[str, object]:
    """Record metadata for a backup dataset without downloading it."""
    if spec.status != "backup":
        raise ValueError(f"{spec.name} is not marked as backup")
    validate_revision(spec.revision)
    result = DownloadResult(
        dataset=spec.name,
        repo_id=spec.repo_id,
        revision=spec.revision,
        license=spec.license,
        status=spec.status,
        path="",
        sha256=None,
        size_bytes=0,
        downloaded_at_utc=utc_now(),
        duration_seconds=0.0,
        extra={"note": "backup only; not downloaded"},
    )
    return update_dataset_versions(manifest_path, result)


def verify_downloads(manifest_path: Path, raw_root: Path) -> list[str]:
    """Return a list of problems for already downloaded datasets."""
    if not manifest_path.exists():
        return [f"missing manifest: {manifest_path}"]
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    for name, entry in data.items():
        if entry.get("status") == "backup" and not entry.get("path"):
            continue
        path = Path(entry["path"])
        if not path.exists():
            problems.append(f"{name}: path does not exist: {path}")
            continue
        archive_sha = entry.get("sha256")
        archive = entry.get("extra", {}).get("archive")
        if archive and archive_sha:
            archive_path = raw_root / name / archive
            if not archive_path.exists():
                problems.append(f"{name}: missing archive: {archive_path}")
            elif sha256_file(archive_path) != archive_sha:
                problems.append(f"{name}: SHA256 mismatch for {archive_path}")
    return problems
