from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from computer_configuration import (  # noqa: E402
    resolve_device,
    resolve_worker_count,
)
from data_loading import build_data_loaders, resolve_dataset_path  # noqa: E402
from documentation_storage import (  # noqa: E402
    RunDocumentationInputs,
    generate_documentation,
)
from leaderboard import DEFAULT_HEATMAP_METRICS  # noqa: E402
from leaderboard import regenerate_all as regenerate_leaderboard  # noqa: E402
from manifest import build_manifest, now_iso  # noqa: E402
from metric_computation import (  # noqa: E402
    compute_aggregate_metrics,
    compute_bootstrap_intervals,
    compute_error_analysis,
)
from models._base import EvaluationResult, TrainingHistory  # noqa: E402
from models.registry import build_adapter, get_model_info  # noqa: E402
from run_storage import (  # noqa: E402
    append_run_to_jsonl,
    build_runs_jsonl_record,
    create_run_directory,
    dump_effective_config,
    dump_manifest,
    dump_metrics,
    dump_misclassified_csv,
    dump_misclassified_samples,
    dump_predictions_csv,
    setup_run_logger,
)

from dataset.input_output import read_dataset  # noqa: E402


@dataclass
class ModelSpec:
    name: str
    factory_kwargs: dict[str, Any]


@dataclass
class RunResult:
    model_name: str
    dataset_name: str
    run_directory: Path
    documentation_directory: Path
    evaluation: EvaluationResult
    history: TrainingHistory


def run_pipeline(
    model_specs: list[ModelSpec],
    dataset_names: list[str],
    configuration: dict[str, Any],
    computer_configuration: dict[str, Any],
    on_run_complete: "Callable[[RunResult], None] | None" = None,
) -> list[RunResult]:
    paths = configuration.get("paths", {})
    raw_directory = PROJECT_ROOT / paths.get("raw_dataset_directory", "dataset/raw")
    processed_directory = PROJECT_ROOT / paths.get(
        "processed_dataset_directory", "dataset/processed"
    )
    runs_root = PROJECT_ROOT / paths.get(
        "machine_learning_run_directory", "machine-learning/runs"
    )
    docs_root = PROJECT_ROOT / paths.get("documentation_directory", "docs")

    training = configuration.get("training", {})
    image_size = int(training.get("image_size", 224))
    batch_size = int(training.get("batch_size", 32))
    random_seed = int(training.get("random_seed", configuration.get("random_seed", 42)))

    device = resolve_device(computer_configuration)
    num_workers = resolve_worker_count(computer_configuration)
    pin_memory = device.type == "cuda"

    class_names = list(configuration.get("class_names") or [])
    results: list[RunResult] = []

    for dataset_name in dataset_names:
        dataset_path = resolve_dataset_path(
            dataset_name, raw_directory, processed_directory
        )
        images, labels = read_dataset(dataset_path)
        splits = build_data_loaders(
            images=images,
            labels=labels,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            random_seed=random_seed,
            pin_memory=pin_memory,
        )

        for model_spec in model_specs:
            run_result = _run_single(
                model_spec=model_spec,
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                splits=splits,
                configuration=configuration,
                computer_configuration=computer_configuration,
                runs_root=runs_root,
                docs_root=docs_root,
                class_names=class_names,
            )
            results.append(run_result)
            if on_run_complete is not None:
                on_run_complete(run_result)

    _rebuild_leaderboard(configuration, docs_root)
    return results


def _rebuild_leaderboard(configuration: dict[str, Any], docs_root: Path) -> None:
    leaderboard_section = configuration.get("leaderboard") or {}
    jsonl_filename = leaderboard_section.get("jsonl_filename", "runs.jsonl")
    jsonl_path = docs_root / jsonl_filename
    if not jsonl_path.exists():
        return
    regenerate_leaderboard(
        jsonl_path=jsonl_path,
        docs_root=docs_root,
        primary_metric=leaderboard_section.get("primary_metric", "balanced_accuracy"),
        secondary_metric=leaderboard_section.get("secondary_metric", "accuracy"),
        heatmap_metrics=list(
            leaderboard_section.get("heatmap_metrics") or DEFAULT_HEATMAP_METRICS
        ),
    )


def _run_single(
    model_spec: ModelSpec,
    dataset_name: str,
    dataset_path: Path,
    splits: Any,
    configuration: dict[str, Any],
    computer_configuration: dict[str, Any],
    runs_root: Path,
    docs_root: Path,
    class_names: list[str],
) -> RunResult:
    info = get_model_info(model_spec.name)
    run_directory = create_run_directory(runs_root, model_spec.name, dataset_name)
    logger = setup_run_logger(run_directory, name=f"{model_spec.name}.{dataset_name}")
    logger.info("run directory: %s", run_directory)
    logger.info("dataset: %s", dataset_name)
    logger.info("model: %s (%s)", info.name, info.display_name)
    started_at = now_iso()

    effective_configuration = {
        **configuration,
        "active_run": {
            "model_name": model_spec.name,
            "dataset_name": dataset_name,
            "factory_kwargs": model_spec.factory_kwargs,
        },
        "computer_configuration": computer_configuration,
    }
    dump_effective_config(run_directory, effective_configuration)

    adapter = build_adapter(model_spec.name, **model_spec.factory_kwargs)
    history = adapter.fit(
        splits=splits,
        configuration=configuration,
        computer_configuration=computer_configuration,
        run_directory=run_directory,
        logger=logger,
    )
    evaluation = adapter.evaluate(
        splits=splits,
        computer_configuration=computer_configuration,
        logger=logger,
    )

    extension = "pth" if info.is_deep_learning else "joblib"
    checkpoint_path = run_directory / f"{run_directory.name}.{extension}"
    adapter.save_checkpoint(checkpoint_path)
    logger.info("checkpoint saved to %s", checkpoint_path)

    evaluation_section = configuration.get("evaluation") or {}
    top_k_values = list(evaluation_section.get("top_k_values") or [3, 5])
    samples_per_pair = int(
        evaluation_section.get("misclassified_samples_per_pair", 10)
    )

    aggregate_metrics = compute_aggregate_metrics(
        targets=evaluation.targets,
        predictions=evaluation.predictions,
        probabilities=evaluation.probabilities,
        num_classes=splits.num_classes,
        is_deep_learning=info.is_deep_learning,
        top_k_values=top_k_values,
    )

    bootstrap_resamples = int(evaluation_section.get("bootstrap_resamples", 1000))
    bootstrap_confidence = float(
        evaluation_section.get("bootstrap_confidence", 0.95)
    )
    if bootstrap_resamples > 0:
        bootstrap_ci = compute_bootstrap_intervals(
            targets=evaluation.targets,
            predictions=evaluation.predictions,
            probabilities=evaluation.probabilities,
            num_classes=splits.num_classes,
            num_resamples=bootstrap_resamples,
            confidence=bootstrap_confidence,
            random_seed=int(
                configuration.get("training", {}).get(
                    "random_seed", configuration.get("random_seed", 42)
                )
            ),
        )
        aggregate_metrics["bootstrap_ci"] = bootstrap_ci

    error_analysis = compute_error_analysis(
        targets=evaluation.targets,
        predictions=evaluation.predictions,
        probabilities=evaluation.probabilities,
        num_classes=splits.num_classes,
    )

    logger.info(
        "metrics: accuracy=%.4f balanced_accuracy=%.4f kappa=%.4f mcc=%.4f",
        aggregate_metrics["accuracy"],
        aggregate_metrics["balanced_accuracy"],
        aggregate_metrics["cohen_kappa"],
        aggregate_metrics["matthews_correlation_coefficient"],
    )
    if aggregate_metrics.get("roc_auc_macro") is not None:
        logger.info(
            "roc_auc_macro=%.4f average_precision_macro=%.4f log_loss=%s",
            aggregate_metrics["roc_auc_macro"],
            aggregate_metrics["average_precision_macro"],
            f"{aggregate_metrics['log_loss']:.4f}"
            if aggregate_metrics.get("log_loss") is not None
            else "n/a",
        )

    metrics_payload = {
        **aggregate_metrics,
        "confusion_matrix": error_analysis.confusion_matrix,
        "confusion_matrix_normalized": error_analysis.confusion_matrix_normalized,
        "per_class_accuracy": error_analysis.per_class_accuracy,
        "most_confused_pairs": error_analysis.most_confused_pairs,
        "misclassified_count": len(error_analysis.misclassified),
        "history": {
            "epochs": history.epochs,
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
            "train_accuracy": history.train_accuracy,
            "val_accuracy": history.val_accuracy,
            "early_stopped": history.early_stopped,
        },
    }
    dump_metrics(run_directory, metrics_payload)

    dump_predictions_csv(
        run_directory=run_directory,
        targets=evaluation.targets,
        predictions=evaluation.predictions,
        probabilities=evaluation.probabilities,
        class_names=class_names,
        num_classes=splits.num_classes,
    )
    dump_misclassified_csv(
        run_directory=run_directory,
        error_analysis=error_analysis,
        class_names=class_names,
        num_classes=splits.num_classes,
    )
    written_pngs = dump_misclassified_samples(
        run_directory=run_directory,
        error_analysis=error_analysis,
        splits=splits,
        class_names=class_names,
        samples_per_pair=samples_per_pair,
    )
    logger.info(
        "wrote predictions.csv, misclassified.csv, and %d misclassified PNGs",
        written_pngs,
    )

    documentation_directory = generate_documentation(
        documentation_root=docs_root,
        inputs=RunDocumentationInputs(
            model_name=model_spec.name,
            model_display_name=info.display_name,
            dataset_name=dataset_name,
            run_directory=run_directory,
            history=history,
            evaluation=evaluation,
            splits=splits,
            class_names=class_names,
            configuration=configuration,
            is_deep_learning=info.is_deep_learning,
            aggregate_metrics=aggregate_metrics,
            error_analysis=error_analysis,
        ),
    )
    logger.info("documentation written to %s", documentation_directory)

    completed_at = now_iso()
    manifest = build_manifest(
        project_root=PROJECT_ROOT,
        dataset_path=dataset_path,
        started_at=started_at,
        completed_at=completed_at,
    )
    dump_manifest(run_directory, manifest)

    leaderboard_section = configuration.get("leaderboard") or {}
    jsonl_filename = leaderboard_section.get("jsonl_filename", "runs.jsonl")
    jsonl_path = docs_root / jsonl_filename
    record = build_runs_jsonl_record(
        run_directory=run_directory,
        documentation_directory=documentation_directory,
        model_name=model_spec.name,
        dataset_name=dataset_name,
        is_deep_learning=info.is_deep_learning,
        aggregate_metrics=aggregate_metrics,
        error_analysis=error_analysis,
        history_epochs=len(history.epochs),
        history_early_stopped=history.early_stopped,
        total_test_samples=int(len(evaluation.targets)),
        manifest=manifest,
    )
    append_run_to_jsonl(jsonl_path, record)
    logger.info(
        "manifest + jsonl appended (duration=%.1fs git=%s)",
        manifest.duration_seconds,
        (manifest.git_commit_sha or "n/a")[:8],
    )

    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    return RunResult(
        model_name=model_spec.name,
        dataset_name=dataset_name,
        run_directory=run_directory,
        documentation_directory=documentation_directory,
        evaluation=evaluation,
        history=history,
    )
