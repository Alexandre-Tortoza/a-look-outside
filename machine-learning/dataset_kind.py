from __future__ import annotations

PROCESSED_DATASET_MARKERS = (
    "_smote",
    "_random_over_sampling",
    "_random_under_sampling",
    "_augmentation_",
)


def dataset_name_is_processed(dataset_name: str) -> bool:
    return any(marker in dataset_name for marker in PROCESSED_DATASET_MARKERS)
