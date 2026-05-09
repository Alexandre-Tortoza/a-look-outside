from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from data_loading import DatasetSplits  # type: ignore[import-not-found]
from manifest import RunManifest  # type: ignore[import-not-found]
from metric_computation import (  # type: ignore[import-not-found]
    ErrorAnalysis,
    MisclassifiedRecord,
)
from PIL import Image


def create_run_directory(
    base_directory: Path,
    model_name: str,
    dataset_name: str,
    today: date | None = None,
) -> Path:
    today = today or date.today()
    base_name = f"{model_name}-{dataset_name}-{today.strftime('%d-%m-%Y')}"
    base_directory.mkdir(parents=True, exist_ok=True)

    candidate = base_directory / base_name
    if not candidate.exists():
        candidate.mkdir(parents=True)
        return candidate

    suffix = 2
    while True:
        candidate = base_directory / f"{base_name}-{suffix}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        suffix += 1


def setup_run_logger(run_directory: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(run_directory / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def dump_effective_config(run_directory: Path, configuration: dict[str, Any]) -> Path:
    output_path = run_directory / "config.yaml"
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(configuration, handle, sort_keys=False)
    return output_path


def dump_metrics(run_directory: Path, metrics: dict[str, Any]) -> Path:
    output_path = run_directory / "metrics.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, default=_json_default)
    return output_path


def dump_manifest(run_directory: Path, manifest: RunManifest) -> Path:
    output_path = run_directory / "manifest.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest.to_dict(), handle, indent=2, default=_json_default)
    return output_path


def append_run_to_jsonl(jsonl_path: Path, record: dict[str, Any]) -> Path:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=_json_default))
        handle.write("\n")
    return jsonl_path


def build_runs_jsonl_record(
    *,
    run_directory: Path,
    documentation_directory: Path,
    model_name: str,
    dataset_name: str,
    is_deep_learning: bool,
    aggregate_metrics: dict[str, Any],
    error_analysis: ErrorAnalysis,
    history_epochs: int,
    history_early_stopped: bool,
    total_test_samples: int,
    manifest: RunManifest,
) -> dict[str, Any]:
    top_k = aggregate_metrics.get("top_k_accuracy") or {}
    return {
        "run_directory_name": run_directory.name,
        "run_directory_path": str(run_directory),
        "documentation_directory": str(documentation_directory),
        "model_name": model_name,
        "dataset_name": dataset_name,
        "is_deep_learning": is_deep_learning,
        "started_at": manifest.started_at,
        "completed_at": manifest.completed_at,
        "duration_seconds": manifest.duration_seconds,
        "date_iso": manifest.completed_at.split("T", 1)[0],
        "accuracy": aggregate_metrics.get("accuracy"),
        "balanced_accuracy": aggregate_metrics.get("balanced_accuracy"),
        "macro_f1": aggregate_metrics.get("f1_macro"),
        "weighted_f1": aggregate_metrics.get("f1_weighted"),
        "micro_f1": aggregate_metrics.get("f1_micro"),
        "precision_macro": aggregate_metrics.get("precision_macro"),
        "recall_macro": aggregate_metrics.get("recall_macro"),
        "cohen_kappa": aggregate_metrics.get("cohen_kappa"),
        "matthews_correlation_coefficient": aggregate_metrics.get(
            "matthews_correlation_coefficient"
        ),
        "log_loss": aggregate_metrics.get("log_loss"),
        "roc_auc_macro": aggregate_metrics.get("roc_auc_macro"),
        "average_precision_macro": aggregate_metrics.get("average_precision_macro"),
        "top_3_accuracy": top_k.get("top_3"),
        "top_5_accuracy": top_k.get("top_5"),
        "total_test_samples": int(total_test_samples),
        "misclassified_count": len(error_analysis.misclassified),
        "epoch_count": int(history_epochs),
        "early_stopped": bool(history_early_stopped),
        "git_commit_sha": manifest.git_commit_sha,
        "git_branch": manifest.git_branch,
        "git_is_dirty": manifest.git_is_dirty,
        "dataset_path": manifest.dataset_path,
        "dataset_sha256": manifest.dataset_sha256,
        "dataset_size_bytes": manifest.dataset_size_bytes,
        "uv_lock_sha256": manifest.uv_lock_sha256,
    }


def dump_predictions_csv(
    run_directory: Path,
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    class_names: list[str],
    num_classes: int,
) -> Path:
    output_path = run_directory / "predictions.csv"
    headers = _prediction_headers(num_classes, class_names)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for test_index in range(len(targets)):
            writer.writerow(
                _prediction_row(
                    test_index=test_index,
                    true_label=int(targets[test_index]),
                    predicted_label=int(predictions[test_index]),
                    probabilities=probabilities,
                    class_names=class_names,
                    num_classes=num_classes,
                )
            )
    return output_path


def dump_misclassified_csv(
    run_directory: Path,
    error_analysis: ErrorAnalysis,
    class_names: list[str],
    num_classes: int,
) -> Path:
    output_path = run_directory / "misclassified.csv"
    headers = _prediction_headers(num_classes, class_names)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for record in error_analysis.misclassified:
            writer.writerow(
                _prediction_row(
                    test_index=record.test_index,
                    true_label=record.true_label,
                    predicted_label=record.predicted_label,
                    probabilities=(
                        record.all_probabilities[None, :]
                        if record.all_probabilities is not None
                        else None
                    ),
                    probabilities_row_index=0,
                    class_names=class_names,
                    num_classes=num_classes,
                )
            )
    return output_path


def dump_misclassified_samples(
    run_directory: Path,
    error_analysis: ErrorAnalysis,
    splits: DatasetSplits,
    class_names: list[str],
    samples_per_pair: int,
) -> int:
    if samples_per_pair <= 0:
        return 0

    base_directory = run_directory / "misclassified_samples"
    base_directory.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[int, int], list[MisclassifiedRecord]] = defaultdict(list)
    for record in error_analysis.misclassified:
        key = (record.true_label, record.predicted_label)
        if len(grouped[key]) < samples_per_pair:
            grouped[key].append(record)

    written = 0
    for (true_label, predicted_label), records in grouped.items():
        true_class = _class_label(true_label, class_names)
        predicted_class = _class_label(predicted_label, class_names)
        pair_directory = base_directory / f"{true_class}_to_{predicted_class}"
        pair_directory.mkdir(parents=True, exist_ok=True)

        for sequential_index, record in enumerate(records, start=1):
            image_array = splits.test_images[record.test_index]
            if image_array.dtype != np.uint8:
                image_array = (image_array * 255).clip(0, 255).astype(np.uint8)
            filename = (
                f"{sequential_index:04d}_idx{record.test_index:05d}.png"
            )
            Image.fromarray(image_array).save(pair_directory / filename)
            written += 1
    return written


def _prediction_headers(num_classes: int, class_names: list[str]) -> list[str]:
    base_headers = [
        "test_index",
        "true_label",
        "true_class",
        "predicted_label",
        "predicted_class",
        "predicted_probability",
        "correct",
    ]
    probability_columns = [
        f"prob_{_class_label(class_index, class_names)}"
        for class_index in range(num_classes)
    ]
    return base_headers + probability_columns


def _prediction_row(
    test_index: int,
    true_label: int,
    predicted_label: int,
    probabilities: np.ndarray | None,
    class_names: list[str],
    num_classes: int,
    probabilities_row_index: int | None = None,
) -> list[Any]:
    correct = "true" if true_label == predicted_label else "false"
    predicted_probability: float | str = ""
    probability_values: list[float | str] = ["" for _ in range(num_classes)]

    if probabilities is not None and probabilities.ndim == 2:
        row_index = (
            probabilities_row_index
            if probabilities_row_index is not None
            else test_index
        )
        if row_index < probabilities.shape[0]:
            row = probabilities[row_index]
            if predicted_label < row.shape[0]:
                predicted_probability = float(row[predicted_label])
            for class_index in range(min(num_classes, row.shape[0])):
                probability_values[class_index] = float(row[class_index])

    return [
        test_index,
        true_label,
        _class_label(true_label, class_names),
        predicted_label,
        _class_label(predicted_label, class_names),
        predicted_probability,
        correct,
        *probability_values,
    ]


def _class_label(class_index: int, class_names: list[str]) -> str:
    if 0 <= class_index < len(class_names):
        return class_names[class_index]
    return f"class_{class_index}"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"object of type {type(value)} is not JSON serializable")
