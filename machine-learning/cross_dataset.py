from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassMapping:
    source_dataset: str
    target_dataset: str
    source_to_common: dict[int, int]
    target_to_common: dict[int, int]
    common_class_names: list[str]


COMMON_CLASS_NAMES = [
    "disk_face_on_no_spiral_or_loose",
    "smooth_completely_round",
    "smooth_in_between_round",
    "smooth_cigar_shaped",
    "disk_edge_on_rounded_or_with_bulge",
    "disk_edge_on_no_bulge",
    "disk_face_on_tight_spiral",
    "disk_face_on_loose_spiral",
]

SDSS_TO_COMMON = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 7,
}

DECALS_TO_COMMON = {
    2: 1,
    3: 2,
    4: 3,
    7: 0,
    8: 5,
    9: 4,
    6: 6,
}


def base_dataset_name(dataset_name: str) -> str:
    return dataset_name.replace("-", "_").split("_", 1)[0]


def resolve_cross_dataset_mapping(
    source_dataset: str,
    target_dataset: str,
) -> ClassMapping:
    source_base = base_dataset_name(source_dataset)
    target_base = base_dataset_name(target_dataset)
    mappings = {
        "sdss": SDSS_TO_COMMON,
        "decals": DECALS_TO_COMMON,
    }
    if source_base not in mappings or target_base not in mappings:
        raise ValueError(
            "cross-dataset mapping only supports sdss and decals, got "
            f"{source_dataset!r} -> {target_dataset!r}"
        )
    return ClassMapping(
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        source_to_common=mappings[source_base],
        target_to_common=mappings[target_base],
        common_class_names=COMMON_CLASS_NAMES,
    )


def apply_label_mapping(
    images: np.ndarray,
    labels: np.ndarray,
    mapping: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    labels = labels.astype(np.int64)
    keep_mask = np.isin(labels, list(mapping))
    filtered_images = images[keep_mask]
    filtered_labels = labels[keep_mask]
    remapped = np.asarray([mapping[int(label)] for label in filtered_labels], dtype=np.int64)
    return filtered_images, remapped
