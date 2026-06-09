from __future__ import annotations

PROCESSED_DATASET_MARKERS = (
    "_smote",
    "_random_over_sampling",
    "_random_under_sampling",
    "_augmentation_",
)

DATASET_KIND_NATURAL = "natural"
DATASET_KIND_PROCESSED = "processed"
DATASET_KIND_UNKNOWN = "unknown"


def dataset_name_is_processed(dataset_name: str) -> bool:
    return any(marker in dataset_name for marker in PROCESSED_DATASET_MARKERS)


def normalize_dataset_path(dataset_path: str | None) -> str:
    """Normalize separators and lowercase for stable path-kind checks."""
    return (dataset_path or "").replace("\\", "/").lower()
