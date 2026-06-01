from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from computer_configuration import (  # type: ignore[import-not-found]
    resolve_device,
    use_mixed_precision,
)
from data_loading import DatasetSplits  # type: ignore[import-not-found]
from sklearn.metrics import (
    classification_report,
)
from sklearn.metrics import (
    confusion_matrix as compute_confusion_matrix,
)
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from models._base import EvaluationResult, TrainingHistory


class FocalLoss(nn.Module):
    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        probabilities = torch.softmax(logits.float(), dim=1)
        target_probabilities = probabilities.gather(1, targets[:, None]).squeeze(1)
        focal_weight = (1.0 - target_probabilities).pow(self.gamma)
        return (focal_weight * cross_entropy).mean()


class DeepLearningAdapter(ABC):
    is_deep_learning = True

    def __init__(self) -> None:
        self._model: nn.Module | None = None
        self._device: torch.device | None = None
        self._best_state_dict: dict[str, Any] | None = None
        self._initial_checkpoint: Path | None = None

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def display_name(self) -> str: ...

    @abstractmethod
    def build_model(self, num_classes: int, image_size: int) -> nn.Module: ...

    def fit(
        self,
        splits: DatasetSplits,
        configuration: dict[str, Any],
        computer_configuration: dict[str, Any],
        run_directory: Path,
        logger: logging.Logger,
    ) -> TrainingHistory:
        training = configuration.get("training", {})
        epoch_count = int(training.get("epoch_count", 50))
        early_stopping_patience = int(training.get("early_stopping_patience", 8))
        learning_rate = float(training.get("learning_rate", 1e-4))
        weight_decay = float(training.get("weight_decay", 1e-4))

        device = resolve_device(computer_configuration)
        self._device = device
        amp_enabled = use_mixed_precision(computer_configuration, device)

        model = self.build_model(splits.num_classes, splits.image_size).to(device)
        self._model = model

        if self._initial_checkpoint is not None:
            payload = torch.load(self._initial_checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(payload["state_dict"])
            logger.info("loaded initial checkpoint from %s", self._initial_checkpoint)

        criterion = self.build_criterion(
            splits=splits,
            configuration=configuration,
            device=device,
        )
        optimizer = self.build_optimizer(
            model=model,
            configuration=configuration,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = self.build_scheduler(
            optimizer=optimizer,
            configuration=configuration,
        )
        scaler = GradScaler(device.type, enabled=amp_enabled)

        logger.info(
            "starting training device=%s amp=%s epochs=%d patience=%d",
            device, amp_enabled, epoch_count, early_stopping_patience,
        )

        history = TrainingHistory()
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_state_dict: dict[str, Any] | None = None

        for epoch in range(1, epoch_count + 1):
            train_loss, train_accuracy = self._run_epoch(
                model, splits.train_loader, criterion, device,
                optimizer=optimizer, scaler=scaler, amp_enabled=amp_enabled,
            )
            val_loss, val_accuracy = self._run_epoch(
                model, splits.val_loader, criterion, device,
                optimizer=None, scaler=scaler, amp_enabled=amp_enabled,
            )
            scheduler.step(val_loss)

            history.epochs.append(epoch)
            history.train_loss.append(train_loss)
            history.val_loss.append(val_loss)
            history.train_accuracy.append(train_accuracy)
            history.val_accuracy.append(val_accuracy)

            logger.info(
                "epoch=%d train_loss=%.4f val_loss=%.4f train_acc=%.4f val_acc=%.4f",
                epoch, train_loss, val_loss, train_accuracy, val_accuracy,
            )

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_state_dict = copy.deepcopy(model.state_dict())
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    logger.info("early stopping triggered at epoch %d", epoch)
                    history.early_stopped = True
                    break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            self._best_state_dict = best_state_dict
        else:
            self._best_state_dict = copy.deepcopy(model.state_dict())

        return history

    def build_criterion(
        self,
        splits: DatasetSplits,
        configuration: dict[str, Any],
        device: torch.device,
    ) -> nn.Module:
        training = configuration.get("training") or {}
        class_weights = _class_weights(
            splits=splits,
            class_weighting=str(training.get("class_weighting", "none")),
            device=device,
        )
        label_smoothing = float(training.get("label_smoothing", 0.0))
        loss_name = str(training.get("loss", "cross_entropy"))
        if loss_name == "cross_entropy":
            return nn.CrossEntropyLoss(
                weight=class_weights,
                label_smoothing=label_smoothing,
            )
        if loss_name == "focal":
            return FocalLoss(
                weight=class_weights,
                gamma=float(training.get("focal_gamma", 2.0)),
                label_smoothing=label_smoothing,
            )
        raise ValueError("training.loss must be one of: cross_entropy, focal")

    def build_optimizer(
        self,
        model: nn.Module,
        configuration: dict[str, Any],
        learning_rate: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        configuration: dict[str, Any],
    ) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

    def evaluate(
        self,
        splits: DatasetSplits,
        computer_configuration: dict[str, Any],
        logger: logging.Logger,
    ) -> EvaluationResult:
        if self._model is None or self._device is None:
            raise RuntimeError("call fit() before evaluate()")

        self._model.eval()
        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        probability_chunks: list[np.ndarray] = []
        amp_enabled = use_mixed_precision(computer_configuration, self._device)

        with torch.no_grad():
            for batch_images, batch_labels in splits.test_loader:
                batch_images = batch_images.to(self._device, non_blocking=True)
                with autocast(self._device.type, enabled=amp_enabled):
                    logits = self._model(batch_images)
                probabilities = torch.softmax(logits.float(), dim=1)
                predictions.append(probabilities.argmax(dim=1).cpu().numpy())
                probability_chunks.append(probabilities.cpu().numpy())
                targets.append(batch_labels.numpy())

        all_predictions = np.concatenate(predictions)
        all_targets = np.concatenate(targets)
        all_probabilities = np.concatenate(probability_chunks, axis=0)
        accuracy = float((all_predictions == all_targets).mean())

        report = classification_report(
            all_targets, all_predictions, output_dict=True, zero_division=0
        )
        matrix = compute_confusion_matrix(all_targets, all_predictions)
        logger.info("test accuracy=%.4f", accuracy)

        return EvaluationResult(
            accuracy=accuracy,
            predictions=all_predictions,
            targets=all_targets,
            classification_report=report,
            confusion_matrix=matrix,
            probabilities=all_probabilities,
        )

    def save_checkpoint(self, path: Path) -> None:
        if self._best_state_dict is None:
            raise RuntimeError("no trained weights to save")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_name": self.name, "state_dict": self._best_state_dict},
            path,
        )

    def _run_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer | None,
        scaler: GradScaler,
        amp_enabled: bool,
    ) -> tuple[float, float]:
        is_training = optimizer is not None
        model.train(is_training)

        loss_sum = 0.0
        correct = 0
        total = 0

        context = torch.enable_grad() if is_training else torch.no_grad()
        with context:
            for batch_images, batch_labels in loader:
                batch_images = batch_images.to(device, non_blocking=True)
                batch_labels = batch_labels.to(device, non_blocking=True)

                with autocast(device.type, enabled=amp_enabled):
                    logits = model(batch_images)
                    loss = criterion(logits, batch_labels)

                if is_training:
                    optimizer.zero_grad(set_to_none=True)
                    if amp_enabled:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                batch_size = batch_images.size(0)
                loss_sum += float(loss.detach().cpu()) * batch_size
                correct += int((logits.argmax(dim=1) == batch_labels).sum().cpu())
                total += batch_size

        if total == 0:
            return 0.0, 0.0
        return loss_sum / total, correct / total


def _class_weights(
    *,
    splits: DatasetSplits,
    class_weighting: str,
    device: torch.device,
) -> torch.Tensor | None:
    if class_weighting == "none":
        return None
    counts = torch.bincount(
        torch.as_tensor(splits.train_labels, dtype=torch.long),
        minlength=splits.num_classes,
    ).float()
    counts = counts.clamp_min(1.0)
    if class_weighting == "inverse_frequency":
        weights = counts.sum() / counts
    elif class_weighting == "sqrt_inverse_frequency":
        weights = torch.sqrt(counts.sum() / counts)
    else:
        raise ValueError(
            "training.class_weighting must be one of: "
            "none, inverse_frequency, sqrt_inverse_frequency"
        )
    return (weights / weights.mean()).to(device)
