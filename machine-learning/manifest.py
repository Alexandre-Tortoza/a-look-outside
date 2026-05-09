from __future__ import annotations

import hashlib
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class RunManifest:
    started_at: str
    completed_at: str
    duration_seconds: float
    git_commit_sha: str | None
    git_branch: str | None
    git_is_dirty: bool
    uv_lock_sha256: str | None
    dataset_path: str
    dataset_sha256: str | None
    dataset_size_bytes: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def capture_git_info(project_root: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "git_commit_sha": None,
        "git_branch": None,
        "git_is_dirty": False,
    }
    if not (project_root / ".git").exists():
        return info

    sha = _run_git(["rev-parse", "HEAD"], project_root)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], project_root)
    status = _run_git(["status", "--porcelain"], project_root)

    if sha is not None:
        info["git_commit_sha"] = sha
    if branch is not None and branch != "HEAD":
        info["git_branch"] = branch
    info["git_is_dirty"] = bool(status)
    return info


def compute_file_sha256(path: Path, chunk_size: int = 1 << 20) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def file_size_bytes(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    return path.stat().st_size


def build_manifest(
    *,
    project_root: Path,
    dataset_path: Path,
    started_at: str,
    completed_at: str,
) -> RunManifest:
    git_info = capture_git_info(project_root)
    uv_lock_path = project_root / "uv.lock"
    duration = _duration_seconds(started_at, completed_at)

    return RunManifest(
        started_at=started_at,
        completed_at=completed_at,
        duration_seconds=duration,
        git_commit_sha=git_info["git_commit_sha"],
        git_branch=git_info["git_branch"],
        git_is_dirty=git_info["git_is_dirty"],
        uv_lock_sha256=compute_file_sha256(uv_lock_path),
        dataset_path=str(dataset_path),
        dataset_sha256=compute_file_sha256(dataset_path),
        dataset_size_bytes=file_size_bytes(dataset_path),
    )


def _run_git(arguments: list[str], project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _duration_seconds(started_at: str, completed_at: str) -> float:
    try:
        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(completed_at)
    except ValueError:
        return 0.0
    return (end - start).total_seconds()
