from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)
    epochs: list[int] = field(default_factory=list)
    early_stopped: bool = False


@dataclass
class EvaluationResult:
    accuracy: float
    predictions: np.ndarray
    targets: np.ndarray
    classification_report: dict[str, Any]
    confusion_matrix: np.ndarray
    probabilities: np.ndarray | None = None
    neighbor_indices: np.ndarray | None = None
    neighbor_distances: np.ndarray | None = None


class ModelAdapter(Protocol):
    name: str
    display_name: str
    is_deep_learning: bool

    def fit(
        self,
        splits: Any,
        configuration: dict[str, Any],
        computer_configuration: dict[str, Any],
        run_directory: Path,
        logger: logging.Logger,
    ) -> TrainingHistory: ...

    def evaluate(
        self,
        splits: Any,
        computer_configuration: dict[str, Any],
        logger: logging.Logger,
    ) -> EvaluationResult: ...

    def save_checkpoint(self, path: Path) -> None: ...
