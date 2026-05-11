from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from data_loading import DatasetSplits  # type: ignore[import-not-found]  # noqa: E402
from metric_computation import ErrorAnalysis  # type: ignore[import-not-found]  # noqa: E402
from models._base import EvaluationResult, TrainingHistory  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402
from text_formatting import (
    format_metric as _format_metric,  # type: ignore[import-not-found]  # noqa: E402
)


@dataclass
class RunDocumentationInputs:
    model_name: str
    model_display_name: str
    dataset_name: str
    run_directory: Path
    history: TrainingHistory
    evaluation: EvaluationResult
    splits: DatasetSplits
    class_names: list[str]
    configuration: dict[str, Any]
    is_deep_learning: bool
    aggregate_metrics: dict[str, Any] | None = None
    error_analysis: ErrorAnalysis | None = None


def generate_documentation(
    documentation_root: Path,
    inputs: RunDocumentationInputs,
) -> Path:
    run_label = inputs.run_directory.name
    output_directory = (
        documentation_root / "models" / inputs.model_name / f"{run_label}-{inputs.dataset_name}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    class_labels = _resolve_class_labels(
        inputs.class_names, inputs.evaluation.targets, inputs.evaluation.predictions
    )

    _write_summary(output_directory, inputs)
    _write_metrics_table(output_directory, inputs)
    _write_classification_report(output_directory, inputs, class_labels)
    _plot_confusion_matrix(output_directory, inputs, class_labels)
    _plot_class_distribution(output_directory, inputs, class_labels)
    _plot_per_class_accuracy(output_directory, inputs, class_labels)

    if inputs.error_analysis is not None:
        _plot_confusion_matrix_normalized(output_directory, inputs, class_labels)
        _write_error_analysis(output_directory, inputs, class_labels)

    if inputs.evaluation.probabilities is not None:
        _plot_roc_curves(output_directory, inputs, class_labels)
        _plot_precision_recall_curves(output_directory, inputs, class_labels)

    if inputs.is_deep_learning and inputs.evaluation.probabilities is not None:
        _plot_calibration(output_directory, inputs)

    if inputs.is_deep_learning and inputs.history.epochs:
        _plot_learning_curves(output_directory, inputs.history)

    if (
        inputs.model_name == "k_nearest_neighbors"
        and inputs.evaluation.neighbor_indices is not None
        and inputs.evaluation.neighbor_distances is not None
    ):
        _plot_neighbor_examples(output_directory, inputs)
        _plot_distance_distribution(output_directory, inputs)

    return output_directory


def _resolve_class_labels(
    configured: list[str],
    targets: np.ndarray,
    predictions: np.ndarray,
) -> list[str]:
    seen = sorted(set(np.concatenate([targets, predictions]).tolist()))
    max_index = max(seen) if seen else -1
    if configured and len(configured) > max_index:
        return list(configured)
    return [f"class_{i}" for i in range(max_index + 1)]


def _write_summary(directory: Path, inputs: RunDocumentationInputs) -> None:
    accuracy = inputs.evaluation.accuracy
    report = inputs.evaluation.classification_report
    macro_f1 = report.get("macro avg", {}).get("f1-score", float("nan"))
    weighted_f1 = report.get("weighted avg", {}).get("f1-score", float("nan"))
    training = inputs.configuration.get("training", {})

    aggregate = inputs.aggregate_metrics or {}
    lines = [
        f"# {inputs.model_display_name} on {inputs.dataset_name}",
        "",
        f"- Run directory: `{inputs.run_directory}`",
        f"- Deep learning: {'yes' if inputs.is_deep_learning else 'no'}",
        f"- Train/val/test sizes: "
        f"{len(inputs.splits.train_labels)}/"
        f"{len(inputs.splits.val_labels)}/"
        f"{len(inputs.splits.test_labels)}",
        f"- Image size: {inputs.splits.image_size}",
        f"- Number of classes: {inputs.splits.num_classes}",
        "",
        "## Headline metrics",
        "",
        f"- Test accuracy: **{accuracy:.4f}**",
        f"- Balanced accuracy: **{_format_metric(aggregate.get('balanced_accuracy'))}**",
        f"- Macro F1: **{macro_f1:.4f}**",
        f"- Weighted F1: **{weighted_f1:.4f}**",
        f"- Cohen kappa: **{_format_metric(aggregate.get('cohen_kappa'))}**",
        f"- MCC: **{_format_metric(aggregate.get('matthews_correlation_coefficient'))}**",
        f"- ROC-AUC (macro): **{_format_metric(aggregate.get('roc_auc_macro'))}**",
        f"- Average precision (macro): **"
        f"{_format_metric(aggregate.get('average_precision_macro'))}**",
        f"- Log loss: **{_format_metric(aggregate.get('log_loss'))}**",
    ]
    top_k_section = aggregate.get("top_k_accuracy") or {}
    for key, value in top_k_section.items():
        lines.append(f"- {key.replace('_', '-')} accuracy: **{value:.4f}**")
    if inputs.error_analysis is not None:
        lines.append(
            f"- Misclassified samples: **{len(inputs.error_analysis.misclassified)}**"
        )

    lines.extend([
        "",
        "## Training configuration",
        "",
        f"- Epoch budget: {training.get('epoch_count', 'n/a')}",
        f"- Early stopping patience: {training.get('early_stopping_patience', 'n/a')}",
        f"- Batch size: {training.get('batch_size', 'n/a')}",
        f"- Learning rate: {training.get('learning_rate', 'n/a')}",
        f"- Random seed: {training.get('random_seed', 'n/a')}",
        "",
        "## Artifacts",
        "",
        "- `metrics.md`: aggregate metrics table.",
        "- `classification_report.md`: per-class precision/recall/F1.",
        "- `confusion_matrix.png`, `confusion_matrix_normalized.png`",
        "- `per_class_accuracy.png`",
        "- `class_distribution.png`",
        "- `roc_curves.png`, `precision_recall_curves.png` (when probabilities available)",
        "- `error_analysis.md` + `misclassified_samples/` in run directory",
        "- `predictions.csv`, `misclassified.csv` in run directory",
    ])

    if inputs.is_deep_learning and inputs.history.epochs:
        lines.append("- `learning_curves.png`")
        lines.append("- `calibration_plot.png`")
    elif not inputs.is_deep_learning:
        lines.append("- `learning_curves.png`: N/A para modelo nao treinavel.")

    if inputs.model_name == "k_nearest_neighbors":
        lines.append("- `neighbor_examples.png`")
        lines.append("- `distance_distribution.png`")

    summary_path = directory / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_metrics_table(directory: Path, inputs: RunDocumentationInputs) -> None:
    aggregate = inputs.aggregate_metrics or {}
    sections: list[str] = [
        f"# Metrics — {inputs.model_display_name} on {inputs.dataset_name}\n",
    ]

    headline_rows = [
        ("accuracy", aggregate.get("accuracy", inputs.evaluation.accuracy)),
        ("balanced_accuracy", aggregate.get("balanced_accuracy")),
        ("cohen_kappa", aggregate.get("cohen_kappa")),
        ("matthews_correlation_coefficient",
         aggregate.get("matthews_correlation_coefficient")),
        ("log_loss", aggregate.get("log_loss")),
        ("roc_auc_macro", aggregate.get("roc_auc_macro")),
        ("average_precision_macro", aggregate.get("average_precision_macro")),
    ]
    headline_table = pd.DataFrame(
        [{"metric": name, "value": _format_metric(value)} for name, value in headline_rows]
    )
    sections.append("## Headline\n")
    sections.append(headline_table.to_markdown(index=False) + "\n")

    averaged_rows = []
    for average in ("macro", "weighted", "micro"):
        averaged_rows.append({
            "average": average,
            "precision": _format_metric(aggregate.get(f"precision_{average}")),
            "recall": _format_metric(aggregate.get(f"recall_{average}")),
            "f1_score": _format_metric(aggregate.get(f"f1_{average}")),
        })
    averaged_table = pd.DataFrame(averaged_rows)
    sections.append("## Averaged precision/recall/F1\n")
    sections.append(averaged_table.to_markdown(index=False) + "\n")

    top_k = aggregate.get("top_k_accuracy") or {}
    if top_k:
        top_k_rows = [
            {"metric": key, "value": _format_metric(value)} for key, value in top_k.items()
        ]
        top_k_table = pd.DataFrame(top_k_rows)
        sections.append("## Top-k accuracy\n")
        sections.append(top_k_table.to_markdown(index=False) + "\n")

    roc_per_class = aggregate.get("roc_auc_per_class") or {}
    ap_per_class = aggregate.get("average_precision_per_class") or {}
    if roc_per_class or ap_per_class:
        per_class_rows = []
        max_index = max(
            list(roc_per_class) + list(ap_per_class) + [-1]
        )
        for class_index in range(max_index + 1):
            per_class_rows.append({
                "class_index": class_index,
                "roc_auc": _format_metric(roc_per_class.get(class_index)),
                "average_precision": _format_metric(ap_per_class.get(class_index)),
            })
        per_class_table = pd.DataFrame(per_class_rows)
        sections.append("## Per-class ROC-AUC / AP\n")
        sections.append(per_class_table.to_markdown(index=False) + "\n")

    metrics_path = directory / "metrics.md"
    metrics_path.write_text("\n".join(sections), encoding="utf-8")


def _write_classification_report(
    directory: Path,
    inputs: RunDocumentationInputs,
    class_labels: list[str],
) -> None:
    report = inputs.evaluation.classification_report
    rows = []
    for index, label in enumerate(class_labels):
        bucket = report.get(str(index)) or report.get(index, {})
        if not bucket:
            continue
        rows.append({
            "class": label,
            "precision": bucket.get("precision", float("nan")),
            "recall": bucket.get("recall", float("nan")),
            "f1_score": bucket.get("f1-score", float("nan")),
            "support": bucket.get("support", float("nan")),
        })
    table = pd.DataFrame(rows)
    path = directory / "classification_report.md"
    path.write_text(
        f"# Classification report — {inputs.model_display_name}\n\n"
        + table.to_markdown(index=False, floatfmt=".4f")
        + "\n",
        encoding="utf-8",
    )


def _plot_confusion_matrix(
    directory: Path,
    inputs: RunDocumentationInputs,
    class_labels: list[str],
) -> None:
    matrix = inputs.evaluation.confusion_matrix
    figure_size = (max(6, len(class_labels) * 0.7), max(5, len(class_labels) * 0.6))
    figure, axis = plt.subplots(figsize=figure_size)
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=axis,
        cbar=True,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(f"{inputs.model_display_name} — {inputs.dataset_name}")
    figure.tight_layout()
    figure.savefig(directory / "confusion_matrix.png", dpi=150)
    plt.close(figure)


def _plot_learning_curves(directory: Path, history: TrainingHistory) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.epochs, history.train_loss, label="train")
    axes[0].plot(history.epochs, history.val_loss, label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.epochs, history.train_accuracy, label="train")
    axes[1].plot(history.epochs, history.val_accuracy, label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    figure.tight_layout()
    figure.savefig(directory / "learning_curves.png", dpi=150)
    plt.close(figure)


def _plot_class_distribution(
    directory: Path,
    inputs: RunDocumentationInputs,
    class_labels: list[str],
) -> None:
    splits = inputs.splits
    counts = []
    for split_name, label_array in (
        ("train", splits.train_labels),
        ("val", splits.val_labels),
        ("test", splits.test_labels),
    ):
        unique, count = np.unique(label_array, return_counts=True)
        for index, total in zip(unique, count, strict=True):
            class_index = int(index)
            class_label = (
                class_labels[class_index]
                if class_index < len(class_labels)
                else f"class_{class_index}"
            )
            counts.append({
                "split": split_name,
                "class": class_label,
                "count": int(total),
            })
    frame = pd.DataFrame(counts)

    figure, axis = plt.subplots(figsize=(max(6, len(class_labels) * 0.7), 4))
    sns.barplot(data=frame, x="class", y="count", hue="split", ax=axis)
    axis.set_title(f"Class distribution — {inputs.dataset_name}")
    axis.tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(directory / "class_distribution.png", dpi=150)
    plt.close(figure)


def _plot_neighbor_examples(directory: Path, inputs: RunDocumentationInputs) -> None:
    indices = inputs.evaluation.neighbor_indices
    if indices is None:
        return
    train_images = inputs.splits.train_images
    test_images = inputs.splits.test_images
    sample_count = min(5, len(test_images))
    neighbor_count = indices.shape[1]

    figure, axes = plt.subplots(
        sample_count, neighbor_count + 1,
        figsize=(2 * (neighbor_count + 1), 2 * sample_count),
    )
    if sample_count == 1:
        axes = np.array([axes])

    for row in range(sample_count):
        axes[row, 0].imshow(test_images[row])
        axes[row, 0].set_title("query" if row == 0 else "")
        axes[row, 0].axis("off")
        for column in range(neighbor_count):
            neighbor_index = int(indices[row, column])
            axes[row, column + 1].imshow(train_images[neighbor_index])
            if row == 0:
                axes[row, column + 1].set_title(f"NN-{column + 1}")
            axes[row, column + 1].axis("off")

    figure.tight_layout()
    figure.savefig(directory / "neighbor_examples.png", dpi=150)
    plt.close(figure)


def _plot_distance_distribution(
    directory: Path,
    inputs: RunDocumentationInputs,
) -> None:
    distances = inputs.evaluation.neighbor_distances
    if distances is None:
        return
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.hist(distances.flatten(), bins=40, color="steelblue")
    axis.set_xlabel("Distance")
    axis.set_ylabel("Count")
    axis.set_title("KNN neighbor distance distribution")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(directory / "distance_distribution.png", dpi=150)
    plt.close(figure)


def _plot_confusion_matrix_normalized(
    directory: Path,
    inputs: RunDocumentationInputs,
    class_labels: list[str],
) -> None:
    if inputs.error_analysis is None:
        return
    matrix = inputs.error_analysis.confusion_matrix_normalized
    figure_size = (max(6, len(class_labels) * 0.7), max(5, len(class_labels) * 0.6))
    figure, axis = plt.subplots(figsize=figure_size)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        xticklabels=class_labels,
        yticklabels=class_labels,
        ax=axis,
        cbar=True,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title(
        f"{inputs.model_display_name} — {inputs.dataset_name} (row-normalized)"
    )
    figure.tight_layout()
    figure.savefig(directory / "confusion_matrix_normalized.png", dpi=150)
    plt.close(figure)


def _plot_per_class_accuracy(
    directory: Path,
    inputs: RunDocumentationInputs,
    class_labels: list[str],
) -> None:
    if inputs.error_analysis is None:
        return
    accuracy_by_class = inputs.error_analysis.per_class_accuracy
    if not accuracy_by_class:
        return
    rows = sorted(
        accuracy_by_class.items(), key=lambda item: item[1], reverse=True
    )
    labels = [
        class_labels[index] if index < len(class_labels) else f"class_{index}"
        for index, _ in rows
    ]
    values = [value for _, value in rows]

    figure, axis = plt.subplots(figsize=(max(6, len(rows) * 0.5), 4))
    axis.barh(labels, values, color="steelblue")
    overall = inputs.aggregate_metrics.get("accuracy") if inputs.aggregate_metrics else None
    if overall is not None:
        axis.axvline(overall, color="firebrick", linestyle="--", label=f"overall={overall:.3f}")
        axis.legend(loc="lower right")
    axis.set_xlim(0, 1)
    axis.set_xlabel("Accuracy")
    axis.set_title(f"Per-class accuracy — {inputs.dataset_name}")
    axis.invert_yaxis()
    axis.grid(True, axis="x", alpha=0.3)
    figure.tight_layout()
    figure.savefig(directory / "per_class_accuracy.png", dpi=150)
    plt.close(figure)


def _plot_roc_curves(
    directory: Path,
    inputs: RunDocumentationInputs,
    class_labels: list[str],
) -> None:
    probabilities = inputs.evaluation.probabilities
    if probabilities is None:
        return
    targets = inputs.evaluation.targets
    aggregate = inputs.aggregate_metrics or {}
    roc_per_class = aggregate.get("roc_auc_per_class") or {}

    figure, axis = plt.subplots(figsize=(7, 6))
    for class_index in range(probabilities.shape[1]):
        binary_targets = (targets == class_index).astype(np.int64)
        if binary_targets.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(binary_targets, probabilities[:, class_index])
        label_text = (
            class_labels[class_index]
            if class_index < len(class_labels)
            else f"class_{class_index}"
        )
        auc_value = roc_per_class.get(class_index)
        legend = (
            f"{label_text} (AUC={auc_value:.3f})"
            if auc_value is not None
            else label_text
        )
        axis.plot(fpr, tpr, label=legend)
    axis.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title(f"ROC curves — {inputs.dataset_name}")
    axis.legend(loc="lower right", fontsize="small")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(directory / "roc_curves.png", dpi=150)
    plt.close(figure)


def _plot_precision_recall_curves(
    directory: Path,
    inputs: RunDocumentationInputs,
    class_labels: list[str],
) -> None:
    probabilities = inputs.evaluation.probabilities
    if probabilities is None:
        return
    targets = inputs.evaluation.targets
    aggregate = inputs.aggregate_metrics or {}
    ap_per_class = aggregate.get("average_precision_per_class") or {}

    figure, axis = plt.subplots(figsize=(7, 6))
    for class_index in range(probabilities.shape[1]):
        binary_targets = (targets == class_index).astype(np.int64)
        if binary_targets.sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(
            binary_targets, probabilities[:, class_index]
        )
        label_text = (
            class_labels[class_index]
            if class_index < len(class_labels)
            else f"class_{class_index}"
        )
        ap_value = ap_per_class.get(class_index)
        legend = (
            f"{label_text} (AP={ap_value:.3f})"
            if ap_value is not None
            else label_text
        )
        axis.plot(recall, precision, label=legend)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(f"Precision-Recall curves — {inputs.dataset_name}")
    axis.legend(loc="lower left", fontsize="small")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(directory / "precision_recall_curves.png", dpi=150)
    plt.close(figure)


def _plot_calibration(directory: Path, inputs: RunDocumentationInputs) -> None:
    probabilities = inputs.evaluation.probabilities
    if probabilities is None:
        return
    targets = inputs.evaluation.targets
    predictions = inputs.evaluation.predictions

    bin_count = int(
        (inputs.configuration.get("evaluation") or {}).get("calibration_bin_count", 10)
    )
    confidences = probabilities[np.arange(len(predictions)), predictions]
    correct = (predictions == targets).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, bin_count + 1)
    bin_indices = np.clip(np.digitize(confidences, bin_edges, right=True) - 1, 0, bin_count - 1)

    bin_confidence: list[float] = []
    bin_accuracy: list[float] = []
    bin_count_per: list[int] = []
    for bucket in range(bin_count):
        mask = bin_indices == bucket
        if not mask.any():
            continue
        bin_confidence.append(float(confidences[mask].mean()))
        bin_accuracy.append(float(correct[mask].mean()))
        bin_count_per.append(int(mask.sum()))

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6, label="perfect calibration")
    if bin_confidence:
        axis.plot(bin_confidence, bin_accuracy, "o-", color="steelblue", label="model")
        for x_value, y_value, count in zip(
            bin_confidence, bin_accuracy, bin_count_per, strict=True
        ):
            axis.annotate(
                f"n={count}", (x_value, y_value),
                textcoords="offset points", xytext=(4, 4), fontsize=7,
            )
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Predicted confidence")
    axis.set_ylabel("Empirical accuracy")
    axis.set_title("Calibration / reliability diagram")
    axis.legend(loc="lower right")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(directory / "calibration_plot.png", dpi=150)
    plt.close(figure)


def _write_error_analysis(
    directory: Path,
    inputs: RunDocumentationInputs,
    class_labels: list[str],
) -> None:
    error_analysis = inputs.error_analysis
    if error_analysis is None:
        return

    sections: list[str] = [
        f"# Error analysis — {inputs.model_display_name} on {inputs.dataset_name}\n",
        "## Per-class accuracy\n",
    ]

    accuracy_rows = []
    for class_index, accuracy_value in sorted(
        error_analysis.per_class_accuracy.items(), key=lambda item: item[1]
    ):
        label_text = (
            class_labels[class_index]
            if class_index < len(class_labels)
            else f"class_{class_index}"
        )
        accuracy_rows.append({
            "class": label_text,
            "accuracy": _format_metric(accuracy_value),
        })
    sections.append(
        pd.DataFrame(accuracy_rows).to_markdown(index=False) + "\n"
    )

    sections.append("## Most-confused pairs (true → predicted)\n")
    if not error_analysis.most_confused_pairs:
        sections.append("_no errors observed_\n")
    else:
        confusion_rows = []
        for true_index, predicted_index, count in error_analysis.most_confused_pairs:
            true_label = (
                class_labels[true_index]
                if true_index < len(class_labels)
                else f"class_{true_index}"
            )
            predicted_label = (
                class_labels[predicted_index]
                if predicted_index < len(class_labels)
                else f"class_{predicted_index}"
            )
            confusion_rows.append({
                "true_class": true_label,
                "predicted_class": predicted_label,
                "count": count,
            })
        sections.append(
            pd.DataFrame(confusion_rows).to_markdown(index=False) + "\n"
        )

    total_errors = len(error_analysis.misclassified)
    sections.append("## Summary\n")
    sections.append(f"- Total misclassified samples: **{total_errors}**\n")
    sections.append(
        f"- Misclassified samples saved at "
        f"`{inputs.run_directory.name}/misclassified_samples/`\n"
    )
    sections.append(
        f"- Full predictions: `{inputs.run_directory.name}/predictions.csv`\n"
    )
    sections.append(
        f"- Errors only: `{inputs.run_directory.name}/misclassified.csv`\n"
    )

    output_path = directory / "error_analysis.md"
    output_path.write_text("\n".join(sections), encoding="utf-8")
