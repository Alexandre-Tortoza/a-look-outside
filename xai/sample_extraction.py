from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MACHINE_LEARNING_ROOT = PROJECT_ROOT / "machine-learning"
for path in (str(PROJECT_ROOT), str(MACHINE_LEARNING_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from data_loading import (  # noqa: E402
    StratifiedSplit,
    resolve_dataset_path,
    stratified_train_val_test_split,
)

from dataset.input_output import read_dataset  # noqa: E402
from xai.artifact_storage import (  # noqa: E402
    class_label,
    sample_filename,
)


@dataclass
class SampleRecord:
    sample_id: str
    class_index: int
    class_label: str
    image_array: np.ndarray
    sample_path: Path
    test_index: int


@dataclass
class ExtractedSamples:
    records: list[SampleRecord]
    split: StratifiedSplit

    def by_class(self) -> dict[int, list[SampleRecord]]:
        grouped: dict[int, list[SampleRecord]] = defaultdict(list)
        for record in self.records:
            grouped[record.class_index].append(record)
        return grouped


def extract_test_samples(
    run_config: dict[str, Any],
    samples_root: Path,
    class_names: list[str],
    project_root: Path,
) -> ExtractedSamples:
    paths = run_config.get("paths") or {}
    raw_directory = project_root / paths.get("raw_dataset_directory", "dataset/raw")
    processed_directory = project_root / paths.get(
        "processed_dataset_directory", "dataset/processed"
    )

    active_run = run_config.get("active_run") or {}
    dataset_name = active_run["dataset_name"]
    training = run_config.get("training") or {}
    random_seed = int(training.get("random_seed", run_config.get("random_seed", 42)))

    dataset_path = resolve_dataset_path(dataset_name, raw_directory, processed_directory)
    images, labels = read_dataset(dataset_path)
    split = stratified_train_val_test_split(images, labels, random_seed)

    counters: dict[int, int] = defaultdict(int)
    records: list[SampleRecord] = []
    for test_index in range(len(split.test_labels)):
        class_index = int(split.test_labels[test_index])
        counters[class_index] += 1
        sequential = counters[class_index]
        label_text = class_label(class_index, class_names)
        filename = sample_filename(label_text, sequential)

        class_directory = samples_root / label_text
        class_directory.mkdir(parents=True, exist_ok=True)
        sample_path = class_directory / filename

        image_array = split.test_images[test_index]
        if image_array.dtype != np.uint8:
            image_array_uint8 = (image_array * 255).clip(0, 255).astype(np.uint8)
        else:
            image_array_uint8 = image_array
        Image.fromarray(image_array_uint8).save(sample_path)

        records.append(
            SampleRecord(
                sample_id=Path(filename).stem,
                class_index=class_index,
                class_label=label_text,
                image_array=image_array_uint8,
                sample_path=sample_path,
                test_index=test_index,
            )
        )

    return ExtractedSamples(records=records, split=split)
