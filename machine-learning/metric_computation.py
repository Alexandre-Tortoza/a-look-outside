from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.metrics import (
    confusion_matrix as compute_confusion_matrix,
)


@dataclass
class MisclassifiedRecord:
    test_index: int
    true_label: int
    predicted_label: int
    predicted_probability: float
    all_probabilities: np.ndarray | None


@dataclass
class ErrorAnalysis:
    confusion_matrix: np.ndarray
    confusion_matrix_normalized: np.ndarray
    per_class_accuracy: dict[int, float]
    most_confused_pairs: list[tuple[int, int, int]]
    misclassified: list[MisclassifiedRecord]


def compute_aggregate_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    num_classes: int,
    is_deep_learning: bool,
    top_k_values: list[int],
) -> dict[str, Any]:
    targets = np.asarray(targets).astype(np.int64)
    predictions = np.asarray(predictions).astype(np.int64)

    metrics: dict[str, Any] = {
        "accuracy": float((predictions == targets).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(targets, predictions)),
        "cohen_kappa": float(cohen_kappa_score(targets, predictions)),
        "matthews_correlation_coefficient": float(
            matthews_corrcoef(targets, predictions)
        ),
    }

    for average in ("macro", "weighted", "micro"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets, predictions, average=average, zero_division=0
        )
        metrics[f"precision_{average}"] = float(precision)
        metrics[f"recall_{average}"] = float(recall)
        metrics[f"f1_{average}"] = float(f1)

    metrics["classification_report"] = classification_report(
        targets, predictions, output_dict=True, zero_division=0
    )

    metrics["log_loss"] = None
    metrics["roc_auc_macro"] = None
    metrics["roc_auc_per_class"] = None
    metrics["average_precision_macro"] = None
    metrics["average_precision_per_class"] = None
    metrics["top_k_accuracy"] = {}

    if probabilities is not None and probabilities.ndim == 2:
        if probabilities.shape[1] != num_classes:
            probabilities = _pad_probabilities(probabilities, num_classes)

        try:
            metrics["log_loss"] = float(
                log_loss(
                    targets,
                    probabilities,
                    labels=list(range(num_classes)),
                )
            )
        except ValueError:
            metrics["log_loss"] = None

        try:
            roc_per_class = roc_auc_score(
                targets,
                probabilities,
                multi_class="ovr",
                average=None,
                labels=list(range(num_classes)),
            )
            metrics["roc_auc_per_class"] = {
                int(class_index): float(value)
                for class_index, value in enumerate(np.atleast_1d(roc_per_class))
            }
            metrics["roc_auc_macro"] = float(np.mean(np.atleast_1d(roc_per_class)))
        except ValueError:
            metrics["roc_auc_per_class"] = None
            metrics["roc_auc_macro"] = None

        per_class_average_precision: dict[int, float] = {}
        binarized_targets = _binarize(targets, num_classes)
        for class_index in range(num_classes):
            class_probabilities = probabilities[:, class_index]
            class_targets = binarized_targets[:, class_index]
            if class_targets.sum() == 0:
                continue
            per_class_average_precision[int(class_index)] = float(
                average_precision_score(class_targets, class_probabilities)
            )
        if per_class_average_precision:
            metrics["average_precision_per_class"] = per_class_average_precision
            metrics["average_precision_macro"] = float(
                np.mean(list(per_class_average_precision.values()))
            )

        if is_deep_learning:
            top_k_results: dict[str, float] = {}
            for k in top_k_values:
                if k <= 1 or k >= num_classes:
                    continue
                try:
                    top_k_results[f"top_{k}"] = float(
                        top_k_accuracy_score(
                            targets,
                            probabilities,
                            k=k,
                            labels=list(range(num_classes)),
                        )
                    )
                except ValueError:
                    continue
            metrics["top_k_accuracy"] = top_k_results

    return metrics


def compute_bootstrap_intervals(
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    num_classes: int,
    num_resamples: int = 1000,
    confidence: float = 0.95,
    random_seed: int = 42,
) -> dict[str, dict[str, float | None]]:
    targets = np.asarray(targets).astype(np.int64)
    predictions = np.asarray(predictions).astype(np.int64)
    sample_count = len(targets)
    if sample_count == 0 or num_resamples <= 0:
        return {}

    rng = np.random.default_rng(random_seed)
    metric_samples: dict[str, list[float]] = {
        "accuracy": [],
        "balanced_accuracy": [],
        "macro_f1": [],
        "cohen_kappa": [],
        "matthews_correlation_coefficient": [],
        "roc_auc_macro": [],
    }

    has_probabilities = probabilities is not None and probabilities.ndim == 2

    for _ in range(num_resamples):
        indices = rng.integers(0, sample_count, size=sample_count)
        boot_targets = targets[indices]
        boot_predictions = predictions[indices]

        metric_samples["accuracy"].append(
            float((boot_predictions == boot_targets).mean())
        )
        metric_samples["balanced_accuracy"].append(
            float(balanced_accuracy_score(boot_targets, boot_predictions))
        )
        metric_samples["macro_f1"].append(
            float(
                f1_score(
                    boot_targets,
                    boot_predictions,
                    average="macro",
                    zero_division=0,
                )
            )
        )
        metric_samples["cohen_kappa"].append(
            float(cohen_kappa_score(boot_targets, boot_predictions))
        )
        metric_samples["matthews_correlation_coefficient"].append(
            float(matthews_corrcoef(boot_targets, boot_predictions))
        )

        if has_probabilities and len(np.unique(boot_targets)) == num_classes:
            try:
                roc_value = roc_auc_score(
                    boot_targets,
                    probabilities[indices],
                    multi_class="ovr",
                    average="macro",
                    labels=list(range(num_classes)),
                )
                metric_samples["roc_auc_macro"].append(float(roc_value))
            except ValueError:
                pass

    alpha = 1.0 - confidence
    low_percentile = 100.0 * alpha / 2.0
    high_percentile = 100.0 * (1.0 - alpha / 2.0)

    intervals: dict[str, dict[str, float | None]] = {}
    for metric, values in metric_samples.items():
        if not values:
            intervals[metric] = {"low": None, "high": None, "mean": None}
            continue
        array = np.asarray(values, dtype=np.float64)
        intervals[metric] = {
            "low": float(np.percentile(array, low_percentile)),
            "high": float(np.percentile(array, high_percentile)),
            "mean": float(array.mean()),
        }
    return intervals


def compute_error_analysis(
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    num_classes: int,
    most_confused_pair_count: int = 10,
) -> ErrorAnalysis:
    targets = np.asarray(targets).astype(np.int64)
    predictions = np.asarray(predictions).astype(np.int64)

    matrix = compute_confusion_matrix(
        targets, predictions, labels=list(range(num_classes))
    )
    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = np.where(row_sums > 0, matrix / row_sums, 0.0)

    per_class_accuracy: dict[int, float] = {}
    for class_index in range(num_classes):
        denominator = int(row_sums[class_index, 0])
        if denominator == 0:
            continue
        per_class_accuracy[int(class_index)] = float(
            matrix[class_index, class_index] / denominator
        )

    confusion_pairs: Counter[tuple[int, int]] = Counter()
    for true_index in range(num_classes):
        for predicted_index in range(num_classes):
            if true_index == predicted_index:
                continue
            count = int(matrix[true_index, predicted_index])
            if count > 0:
                confusion_pairs[(true_index, predicted_index)] = count

    most_confused_pairs = [
        (int(true_index), int(predicted_index), int(count))
        for (true_index, predicted_index), count in confusion_pairs.most_common(
            most_confused_pair_count
        )
    ]

    misclassified = _collect_misclassified(targets, predictions, probabilities)

    return ErrorAnalysis(
        confusion_matrix=matrix,
        confusion_matrix_normalized=normalized,
        per_class_accuracy=per_class_accuracy,
        most_confused_pairs=most_confused_pairs,
        misclassified=misclassified,
    )


def _collect_misclassified(
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
) -> list[MisclassifiedRecord]:
    error_indices = np.where(targets != predictions)[0]
    records: list[MisclassifiedRecord] = []
    for test_index in error_indices:
        predicted_label = int(predictions[test_index])
        if probabilities is not None and probabilities.ndim == 2:
            row = probabilities[test_index]
            predicted_probability = float(row[predicted_label])
            all_probabilities: np.ndarray | None = row.astype(np.float32)
        else:
            predicted_probability = float("nan")
            all_probabilities = None
        records.append(
            MisclassifiedRecord(
                test_index=int(test_index),
                true_label=int(targets[test_index]),
                predicted_label=predicted_label,
                predicted_probability=predicted_probability,
                all_probabilities=all_probabilities,
            )
        )
    records.sort(key=lambda record: record.predicted_probability, reverse=True)
    return records


def _pad_probabilities(probabilities: np.ndarray, num_classes: int) -> np.ndarray:
    if probabilities.shape[1] >= num_classes:
        return probabilities[:, :num_classes]
    padded = np.zeros((probabilities.shape[0], num_classes), dtype=probabilities.dtype)
    padded[:, : probabilities.shape[1]] = probabilities
    return padded


def _binarize(targets: np.ndarray, num_classes: int) -> np.ndarray:
    binarized = np.zeros((targets.shape[0], num_classes), dtype=np.int64)
    binarized[np.arange(targets.shape[0]), targets] = 1
    return binarized
