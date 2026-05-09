from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CHECKPOINT_EXTENSIONS = (".pth", ".joblib")


@dataclass
class RunArtifact:
    run_directory: Path
    model_name: str
    dataset_name: str
    checkpoint_path: Path
    run_config: dict[str, Any]
    factory_kwargs: dict[str, Any]

    @property
    def display_label(self) -> str:
        return f"{self.model_name} | {self.dataset_name} | {self.run_directory.name}"


def discover_runs(runs_root: Path) -> list[RunArtifact]:
    if not runs_root.exists():
        return []

    artifacts: list[RunArtifact] = []
    for entry in sorted(runs_root.iterdir()):
        if not entry.is_dir():
            continue
        config_path = entry / "config.yaml"
        if not config_path.exists():
            continue

        with config_path.open("r", encoding="utf-8") as handle:
            run_config = yaml.safe_load(handle) or {}

        active_run = run_config.get("active_run") or {}
        model_name = active_run.get("model_name")
        dataset_name = active_run.get("dataset_name")
        if not model_name or not dataset_name:
            continue

        checkpoint_path = _find_checkpoint(entry)
        if checkpoint_path is None:
            continue

        artifacts.append(
            RunArtifact(
                run_directory=entry,
                model_name=model_name,
                dataset_name=dataset_name,
                checkpoint_path=checkpoint_path,
                run_config=run_config,
                factory_kwargs=dict(active_run.get("factory_kwargs") or {}),
            )
        )
    return artifacts


def output_directories(
    xai_root: Path,
    model_name: str,
    dataset_name: str,
) -> tuple[Path, Path]:
    samples_directory = xai_root / model_name / dataset_name / "samples"
    explanations_directory = xai_root / model_name / dataset_name / "xai"
    samples_directory.mkdir(parents=True, exist_ok=True)
    explanations_directory.mkdir(parents=True, exist_ok=True)
    return samples_directory, explanations_directory


def class_label(class_index: int, class_names: list[str]) -> str:
    if 0 <= class_index < len(class_names):
        return class_names[class_index]
    return f"class_{class_index}"


def sample_filename(class_label_value: str, sequential_index: int) -> str:
    return f"{class_label_value}_{sequential_index:04d}.png"


def _find_checkpoint(run_directory: Path) -> Path | None:
    for extension in CHECKPOINT_EXTENSIONS:
        for candidate in run_directory.glob(f"*{extension}"):
            return candidate
    return None
