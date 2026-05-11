from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MACHINE_LEARNING_ROOT = PROJECT_ROOT / "machine-learning"
for path in (str(PROJECT_ROOT), str(MACHINE_LEARNING_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from computer_configuration import (  # noqa: E402
    resolve_device,
    use_mixed_precision,
)
from data_loading import (  # noqa: E402
    GalaxyImageDataset,
    flatten_normalized,
)
from models.k_nearest_neighbors import _build_feature_extractor  # noqa: E402

from dataset.input_output import resolve_class_names  # noqa: E402
from xai.artifact_storage import class_label  # noqa: E402
from xai.sample_extraction import ExtractedSamples  # noqa: E402


def apply(
    *,
    samples: ExtractedSamples,
    explanations_directory: Path,
    run_config: dict[str, Any],
    factory_kwargs: dict[str, Any],
    checkpoint_path: Path,
    computer_configuration: dict[str, Any],
    logger: logging.Logger,
) -> int:
    payload = joblib.load(checkpoint_path)
    classifier = payload["classifier"]
    mode = payload.get("mode", factory_kwargs.get("mode", "pixels"))

    xai_section = (run_config.get("xai") or {}).get("knn") or {}
    panel_neighbor_count = int(xai_section.get("panel_neighbor_count", 5))
    n_neighbors = min(panel_neighbor_count, classifier.n_neighbors)

    training = run_config.get("training") or {}
    image_size = int(training.get("image_size", 224))
    dataset_name = (run_config.get("active_run") or {}).get("dataset_name", "")
    class_names = resolve_class_names(run_config.get("class_names"), dataset_name)
    train_images = samples.split.train_images
    train_labels = samples.split.train_labels

    test_features = _compute_test_features(
        mode=mode,
        test_images=samples.split.test_images,
        image_size=image_size,
        computer_configuration=computer_configuration,
    )

    written = 0
    for record in samples.records:
        query_features = test_features[record.test_index : record.test_index + 1]
        distances, indices = classifier.kneighbors(
            query_features, n_neighbors=n_neighbors
        )
        neighbor_indices = indices[0]
        neighbor_distances = distances[0]

        figure, axes = plt.subplots(
            1, n_neighbors + 1,
            figsize=(2 * (n_neighbors + 1), 2.5),
        )
        axes[0].imshow(record.image_array)
        axes[0].set_title(f"query\n{record.class_label}")
        axes[0].axis("off")
        for column, neighbor_index in enumerate(neighbor_indices):
            neighbor_class = int(train_labels[neighbor_index])
            neighbor_label = class_label(neighbor_class, class_names)
            axes[column + 1].imshow(train_images[neighbor_index])
            axes[column + 1].set_title(
                f"NN-{column + 1}\n{neighbor_label}\nd={neighbor_distances[column]:.2f}"
            )
            axes[column + 1].axis("off")

        class_directory = explanations_directory / record.class_label
        class_directory.mkdir(parents=True, exist_ok=True)
        output_path = class_directory / f"{record.sample_id}.png"
        figure.tight_layout()
        figure.savefig(output_path, dpi=120)
        plt.close(figure)
        written += 1

    logger.info("nearest_neighbors: wrote %d explanation(s)", written)
    return written


def _compute_test_features(
    mode: str,
    test_images: np.ndarray,
    image_size: int,
    computer_configuration: dict[str, Any],
) -> np.ndarray:
    if mode == "pixels":
        return flatten_normalized(test_images)

    device = resolve_device(computer_configuration)
    amp_enabled = use_mixed_precision(computer_configuration, device)

    feature_extractor = _build_feature_extractor().to(device).eval()
    dataset = GalaxyImageDataset(
        test_images, np.zeros(len(test_images), dtype=np.int64), image_size
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch_images, _ in loader:
            batch_images = batch_images.to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                features = feature_extractor(batch_images)
            chunks.append(features.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)
