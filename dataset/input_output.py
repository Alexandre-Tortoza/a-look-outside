from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np

DATASET_IMAGES_KEY = "images"
DATASET_LABELS_KEY = "ans"

DatasetOrigin = Literal["raw", "processed"]


@dataclass(frozen=True)
class DatasetEntry:
    name: str
    path: Path
    origin: DatasetOrigin

    @property
    def display_label(self) -> str:
        return f"[{self.origin}] {self.name}"


def read_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"dataset file not found: {path}")
    with h5py.File(path, "r") as handle:
        if DATASET_IMAGES_KEY not in handle or DATASET_LABELS_KEY not in handle:
            raise KeyError(
                f"{path} is missing required keys "
                f"'{DATASET_IMAGES_KEY}' and/or '{DATASET_LABELS_KEY}'"
            )
        images = handle[DATASET_IMAGES_KEY][...]
        labels = handle[DATASET_LABELS_KEY][...]
    if len(images) != len(labels):
        raise ValueError(
            f"{path}: images ({len(images)}) and labels ({len(labels)}) length mismatch"
        )
    return np.asarray(images), np.asarray(labels)


def write_dataset(path: Path, images: np.ndarray, labels: np.ndarray) -> None:
    if len(images) != len(labels):
        raise ValueError(
            f"images ({len(images)}) and labels ({len(labels)}) length mismatch"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            DATASET_IMAGES_KEY,
            data=images,
            compression="gzip",
            compression_opts=4,
        )
        handle.create_dataset(
            DATASET_LABELS_KEY,
            data=labels,
            compression="gzip",
            compression_opts=4,
        )


def list_available_datasets(
    raw_directory: Path,
    processed_directory: Path,
) -> list[DatasetEntry]:
    entries: list[DatasetEntry] = []
    for directory, origin in ((raw_directory, "raw"), (processed_directory, "processed")):
        if not directory.exists():
            continue
        for file_path in sorted(directory.glob("*.h5")):
            entries.append(
                DatasetEntry(name=file_path.stem, path=file_path, origin=origin)
            )
    return entries


def derive_dataset_label(entry: DatasetEntry) -> str:
    return f"{entry.name}-raw" if entry.origin == "raw" else entry.name


def resolve_class_names(class_names_config: Any, dataset_name: str) -> list[str]:
    """Return the class name list for a given dataset.

    Supports both a flat list (same names for all datasets) and a dict keyed by
    base dataset name (e.g. ``{"sdss": [...], "decals": [...]}``)."""
    if isinstance(class_names_config, list):
        return list(class_names_config)
    if isinstance(class_names_config, dict):
        base = dataset_name.replace("-", "_").split("_")[0]
        return list(class_names_config.get(base) or [])
    return []


def class_distribution(labels: np.ndarray) -> dict[int, int]:
    unique_labels, counts = np.unique(labels, return_counts=True)
    return {
        int(label): int(count)
        for label, count in zip(unique_labels, counts, strict=True)
    }
