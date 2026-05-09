from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import torch
from computer_configuration import (  # type: ignore[import-not-found]
    resolve_device,
    use_mixed_precision,
)
from data_loading import (  # type: ignore[import-not-found]
    DatasetSplits,
    GalaxyImageDataset,
    flatten_normalized,
)
from sklearn.metrics import (
    classification_report,
)
from sklearn.metrics import (
    confusion_matrix as compute_confusion_matrix,
)
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
from torch.utils.data import DataLoader

from models._base import EvaluationResult, TrainingHistory

KnnMode = Literal["pixels", "features"]


def _build_feature_extractor() -> nn.Module:
    from torchvision.models import ResNet50_Weights, resnet50

    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    backbone.fc = nn.Identity()
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    return backbone


class KNearestNeighborsAdapter:
    name = "k_nearest_neighbors"
    display_name = "K-Nearest Neighbors"
    is_deep_learning = False

    def __init__(self, mode: KnnMode = "pixels", n_neighbors: int = 5) -> None:
        self._mode: KnnMode = mode
        self._n_neighbors = int(n_neighbors)
        self._classifier: KNeighborsClassifier | None = None
        self._feature_extractor: nn.Module | None = None
        self._device: torch.device | None = None
        self._train_features: np.ndarray | None = None

    @property
    def display_name_with_mode(self) -> str:
        return f"{self.display_name} ({self._mode})"

    def fit(
        self,
        splits: DatasetSplits,
        configuration: dict[str, Any],
        computer_configuration: dict[str, Any],
        run_directory: Path,
        logger: logging.Logger,
    ) -> TrainingHistory:
        self._device = resolve_device(computer_configuration)

        if self._mode == "pixels":
            train_features = flatten_normalized(splits.train_images)
        else:
            self._feature_extractor = _build_feature_extractor().to(self._device)
            train_features = self._extract_features(
                splits.train_images, splits.image_size, computer_configuration, logger
            )

        logger.info(
            "fitting KNN mode=%s n_neighbors=%d feature_dim=%d",
            self._mode, self._n_neighbors, train_features.shape[1],
        )
        classifier = KNeighborsClassifier(n_neighbors=self._n_neighbors)
        classifier.fit(train_features, splits.train_labels)

        self._classifier = classifier
        self._train_features = train_features
        return TrainingHistory()

    def evaluate(
        self,
        splits: DatasetSplits,
        computer_configuration: dict[str, Any],
        logger: logging.Logger,
    ) -> EvaluationResult:
        if self._classifier is None:
            raise RuntimeError("call fit() before evaluate()")

        if self._mode == "pixels":
            test_features = flatten_normalized(splits.test_images)
        else:
            test_features = self._extract_features(
                splits.test_images, splits.image_size, computer_configuration, logger
            )

        predictions = self._classifier.predict(test_features)
        probabilities = self._classifier.predict_proba(test_features)
        targets = splits.test_labels.astype(np.int64)
        accuracy = float((predictions == targets).mean())

        report = classification_report(
            targets, predictions, output_dict=True, zero_division=0
        )
        matrix = compute_confusion_matrix(targets, predictions)
        distances, indices = self._classifier.kneighbors(test_features)

        logger.info("test accuracy=%.4f", accuracy)
        return EvaluationResult(
            accuracy=accuracy,
            predictions=predictions,
            targets=targets,
            classification_report=report,
            confusion_matrix=matrix,
            probabilities=probabilities,
            neighbor_indices=indices,
            neighbor_distances=distances,
        )

    def save_checkpoint(self, path: Path) -> None:
        if self._classifier is None:
            raise RuntimeError("nothing to save")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "model_name": self.name,
            "mode": self._mode,
            "n_neighbors": self._n_neighbors,
            "classifier": self._classifier,
            "train_features": self._train_features,
        }
        joblib.dump(payload, path)

    def _extract_features(
        self,
        images: np.ndarray,
        image_size: int,
        computer_configuration: dict[str, Any],
        logger: logging.Logger,
    ) -> np.ndarray:
        if self._feature_extractor is None:
            self._feature_extractor = _build_feature_extractor().to(
                self._device or torch.device("cpu")
            )
        device = self._device or torch.device("cpu")
        amp_enabled = use_mixed_precision(computer_configuration, device)

        dataset = GalaxyImageDataset(
            images, np.zeros(len(images), dtype=np.int64), image_size
        )
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

        chunks: list[np.ndarray] = []
        self._feature_extractor.eval()
        with torch.no_grad():
            for batch_images, _ in loader:
                batch_images = batch_images.to(device, non_blocking=True)
                with torch.amp.autocast(device.type, enabled=amp_enabled):
                    features = self._feature_extractor(batch_images)
                chunks.append(features.float().cpu().numpy())
        return np.concatenate(chunks, axis=0)
