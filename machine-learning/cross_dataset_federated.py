from __future__ import annotations

import copy
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix as compute_confusion_matrix
from torch.amp import GradScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from computer_configuration import (  # noqa: E402
    resolve_device,
    resolve_worker_count,
    use_mixed_precision,
)
from cross_dataset import apply_label_mapping, resolve_cross_dataset_mapping  # noqa: E402
from data_loading import build_data_loaders, resolve_dataset_path  # noqa: E402
from leaderboard import DEFAULT_HEATMAP_METRICS  # noqa: E402
from leaderboard import regenerate_all as regenerate_leaderboard  # noqa: E402
from manifest import build_manifest, now_iso  # noqa: E402
from metric_computation import compute_aggregate_metrics, compute_error_analysis  # noqa: E402
from models._base import EvaluationResult, TrainingHistory  # noqa: E402
from models.dino import DinoAdapter  # noqa: E402
from pipeline import ModelSpec, run_pipeline  # noqa: E402
from run_storage import (  # noqa: E402
    append_run_to_jsonl,
    build_runs_jsonl_record,
    create_run_directory,
    dump_effective_config,
    dump_manifest,
    dump_metrics,
    setup_run_logger,
)

from dataset.input_output import read_dataset  # noqa: E402


@dataclass
class ExperimentResult:
    top_model_names: list[str]
    cross_dataset_runs: list[Path]
    federated_run_directory: Path | None


@dataclass
class LoadedDataset:
    name: str
    path: Path
    splits: Any


def run_experiment(
    benchmark_spec: dict[str, Any],
    config: dict[str, Any],
    computer_configuration: dict[str, Any],
    project_root: Path,
    console: Console,
    logger: logging.Logger,
) -> ExperimentResult:
    paths = config.get("paths") or {}
    raw_directory = project_root / paths.get("raw_dataset_directory", "dataset/raw")
    processed_directory = project_root / paths.get(
        "processed_dataset_directory", "dataset/processed"
    )
    runs_root = project_root / paths.get(
        "machine_learning_run_directory", "machine-learning/runs"
    )
    docs_root = project_root / paths.get("documentation_directory", "docs")

    top_model_names = _resolve_top_model_names(
        benchmark_spec=benchmark_spec,
        config=config,
        computer_configuration=computer_configuration,
        docs_root=docs_root,
        logger=logger,
    )
    console.print(f"[cyan]Top models:[/cyan] {', '.join(top_model_names)}")

    cross_runs = _run_cross_dataset_dino(
        benchmark_spec=benchmark_spec,
        config=config,
        computer_configuration=computer_configuration,
        raw_directory=raw_directory,
        processed_directory=processed_directory,
        runs_root=runs_root,
        docs_root=docs_root,
        logger=logger,
    )

    federated_run = _run_federated_dino(
        benchmark_spec=benchmark_spec,
        config=config,
        computer_configuration=computer_configuration,
        raw_directory=raw_directory,
        processed_directory=processed_directory,
        runs_root=runs_root,
        docs_root=docs_root,
        logger=logger,
    )

    _rebuild_leaderboard(config, docs_root)
    return ExperimentResult(
        top_model_names=top_model_names,
        cross_dataset_runs=cross_runs,
        federated_run_directory=federated_run,
    )


def _resolve_top_model_names(
    *,
    benchmark_spec: dict[str, Any],
    config: dict[str, Any],
    computer_configuration: dict[str, Any],
    docs_root: Path,
    logger: logging.Logger,
) -> list[str]:
    configured_top_models = list(benchmark_spec.get("top_models") or [])
    if configured_top_models:
        return [str(model_name) for model_name in configured_top_models]

    top_count = int(benchmark_spec.get("top_model_count", 3))
    primary_metric = str(benchmark_spec.get("primary_metric", "balanced_accuracy"))
    jsonl_path = docs_root / (config.get("leaderboard") or {}).get(
        "jsonl_filename", "runs.jsonl"
    )
    selected = _top_distinct_models_from_jsonl(jsonl_path, primary_metric, top_count)
    if len(selected) >= top_count:
        return selected[:top_count]

    logger.info(
        "leaderboard only has %d distinct models; running quick top-model benchmark",
        len(selected),
    )
    quick = benchmark_spec.get("quick_top_models") or {}
    candidate_models = list(
        quick.get("candidate_models")
        or ["dino", "efficientnet", "resnet50", "vgg16", "k_nearest_neighbors"]
    )
    dataset_names = list(quick.get("dataset_names") or ["sdss_raw"])
    training_overrides = dict(
        quick.get("training")
        or {"epoch_count": 3, "early_stopping_patience": 2}
    )
    quick_config = copy.deepcopy(config)
    quick_config["training"] = {
        **(quick_config.get("training") or {}),
        **training_overrides,
    }
    quick_config["evaluation"] = {
        **(quick_config.get("evaluation") or {}),
        "bootstrap_resamples": 0,
    }
    model_specs = [
        ModelSpec(
            name=model_name,
            factory_kwargs=(
                {"mode": "pixels", "n_neighbors": 5}
                if model_name == "k_nearest_neighbors"
                else {}
            ),
        )
        for model_name in candidate_models
    ]
    results = run_pipeline(
        model_specs=model_specs,
        dataset_names=dataset_names,
        configuration=quick_config,
        computer_configuration=computer_configuration,
    )
    ranked = sorted(
        results,
        key=lambda result: float(
            getattr(result.evaluation, "accuracy", 0.0)
        ),
        reverse=True,
    )
    top_models: list[str] = []
    for result in ranked:
        if result.model_name not in top_models:
            top_models.append(result.model_name)
        if len(top_models) == top_count:
            break
    return top_models


def _top_distinct_models_from_jsonl(
    jsonl_path: Path,
    primary_metric: str,
    top_count: int,
) -> list[str]:
    if not jsonl_path.exists():
        return []
    records: list[dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get(primary_metric) is not None:
                records.append(record)
    records.sort(key=lambda record: float(record.get(primary_metric) or 0.0), reverse=True)
    selected: list[str] = []
    for record in records:
        model_name = str(record.get("model_name") or "")
        if model_name and model_name not in selected:
            selected.append(model_name)
        if len(selected) == top_count:
            break
    return selected


def _run_cross_dataset_dino(
    *,
    benchmark_spec: dict[str, Any],
    config: dict[str, Any],
    computer_configuration: dict[str, Any],
    raw_directory: Path,
    processed_directory: Path,
    runs_root: Path,
    docs_root: Path,
    logger: logging.Logger,
) -> list[Path]:
    datasets = list(benchmark_spec.get("datasets") or ["sdss_raw", "decals_raw"])
    if len(datasets) != 2:
        raise ValueError("cross-dataset DINO experiment expects exactly two datasets")

    run_directories: list[Path] = []
    for train_name, eval_name in ((datasets[0], datasets[1]), (datasets[1], datasets[0])):
        mapping = resolve_cross_dataset_mapping(train_name, eval_name)
        train_data = _load_dataset(
            train_name,
            mapping.source_to_common,
            config,
            computer_configuration,
            raw_directory,
            processed_directory,
        )
        eval_data = _load_dataset(
            eval_name,
            mapping.target_to_common,
            config,
            computer_configuration,
            raw_directory,
            processed_directory,
        )
        adapter = DinoAdapter(**_dino_kwargs(config))
        run_directory = create_run_directory(
            runs_root, "dino", f"train_{train_name}_eval_{eval_name}"
        )
        run_logger = setup_run_logger(
            run_directory, name=f"dino.train_{train_name}.eval_{eval_name}"
        )
        run_logger.info("cross-dataset training=%s evaluation=%s", train_name, eval_name)
        started_at = now_iso()
        effective_config = {
            **config,
            "active_run": {
                "model_name": "dino",
                "dataset_name": f"train_{train_name}_eval_{eval_name}",
                "train_dataset_name": train_name,
                "evaluation_dataset_name": eval_name,
                "class_mapping": mapping.__dict__,
            },
            "computer_configuration": computer_configuration,
        }
        dump_effective_config(run_directory, effective_config)
        history = adapter.fit(
            splits=train_data.splits,
            configuration=config,
            computer_configuration=computer_configuration,
            run_directory=run_directory,
            logger=run_logger,
        )
        evaluation = adapter.evaluate(
            splits=eval_data.splits,
            computer_configuration=computer_configuration,
            logger=run_logger,
        )
        checkpoint_path = run_directory / f"{run_directory.name}.pth"
        adapter.save_checkpoint(checkpoint_path)
        _persist_evaluation(
            run_directory=run_directory,
            docs_root=docs_root,
            model_name="dino",
            dataset_name=f"train_{train_name}_eval_{eval_name}",
            evaluation=evaluation,
            history=history,
            splits=eval_data.splits,
            class_names=mapping.common_class_names,
            config=config,
            dataset_path=eval_data.path,
            started_at=started_at,
            is_deep_learning=True,
            logger=run_logger,
        )
        _close_logger(run_logger)
        run_directories.append(run_directory)
    return run_directories


def _run_federated_dino(
    *,
    benchmark_spec: dict[str, Any],
    config: dict[str, Any],
    computer_configuration: dict[str, Any],
    raw_directory: Path,
    processed_directory: Path,
    runs_root: Path,
    docs_root: Path,
    logger: logging.Logger,
) -> Path:
    federated = benchmark_spec.get("federated") or {}
    datasets = list(benchmark_spec.get("datasets") or ["sdss_raw", "decals_raw"])
    if len(datasets) != 2:
        raise ValueError("federated DINO experiment expects exactly two datasets")
    rounds = int(federated.get("rounds", 3))
    local_epochs = int(federated.get("local_epochs", 1))

    mapping = resolve_cross_dataset_mapping(datasets[0], datasets[1])
    clients = [
        _load_dataset(
            datasets[0],
            mapping.source_to_common,
            config,
            computer_configuration,
            raw_directory,
            processed_directory,
        ),
        _load_dataset(
            datasets[1],
            mapping.target_to_common,
            config,
            computer_configuration,
            raw_directory,
            processed_directory,
        ),
    ]
    run_directory = create_run_directory(runs_root, "federated_dino", "sdss_decals")
    run_logger = setup_run_logger(run_directory, name="federated_dino.sdss_decals")
    started_at = now_iso()
    dump_effective_config(
        run_directory,
        {
            **config,
            "active_run": {
                "model_name": "federated_dino",
                "dataset_name": "sdss_decals",
                "client_dataset_names": datasets,
                "rounds": rounds,
                "local_epochs": local_epochs,
                "class_mapping": mapping.__dict__,
            },
            "computer_configuration": computer_configuration,
        },
    )

    device = resolve_device(computer_configuration)
    amp_enabled = use_mixed_precision(computer_configuration, device)
    adapter = DinoAdapter(**_dino_kwargs(config))
    global_model = adapter.build_model(
        clients[0].splits.num_classes, clients[0].splits.image_size
    ).to(device)
    latest_client_models: dict[str, torch.nn.Module] = {}
    round_metrics: list[dict[str, Any]] = []

    for round_index in range(1, rounds + 1):
        run_logger.info("federated round %d/%d", round_index, rounds)
        client_states: list[tuple[int, dict[str, torch.Tensor]]] = []
        for client in clients:
            local_adapter = DinoAdapter(**_dino_kwargs(config))
            local_model = local_adapter.build_model(
                client.splits.num_classes, client.splits.image_size
            ).to(device)
            local_model.load_state_dict(global_model.state_dict())
            local_adapter._model = local_model
            local_adapter._device = device
            _train_local_model(
                adapter=local_adapter,
                model=local_model,
                splits=client.splits,
                configuration=config,
                computer_configuration=computer_configuration,
                device=device,
                amp_enabled=amp_enabled,
                local_epochs=local_epochs,
                logger=run_logger,
            )
            latest_client_models[client.name] = local_model
            train_size = int(len(client.splits.train_labels))
            client_states.append((train_size, copy.deepcopy(local_model.state_dict())))

        global_model.load_state_dict(_weighted_average_state_dicts(client_states))
        round_record: dict[str, Any] = {"round": round_index}
        for client in clients:
            evaluation = _evaluate_torch_model(
                global_model,
                client.splits,
                computer_configuration,
                device,
                run_logger,
            )
            metrics = compute_aggregate_metrics(
                targets=evaluation.targets,
                predictions=evaluation.predictions,
                probabilities=evaluation.probabilities,
                num_classes=client.splits.num_classes,
                is_deep_learning=True,
                top_k_values=_top_k_values(config),
            )
            round_record[f"global_on_{client.name}"] = metrics
        round_metrics.append(round_record)
        torch.save(
            {"model_name": "federated_dino", "state_dict": global_model.state_dict()},
            run_directory / f"round_{round_index:03d}.pth",
        )

    final_evaluations: dict[str, Any] = {}
    for client in clients:
        final_evaluations[f"global_on_{client.name}"] = _persist_model_evaluation(
            model=global_model,
            model_name="federated_dino",
            dataset_name=f"global_on_{client.name}",
            dataset_path=client.path,
            splits=client.splits,
            class_names=mapping.common_class_names,
            config=config,
            computer_configuration=computer_configuration,
            run_directory=run_directory,
            docs_root=docs_root,
            started_at=started_at,
            logger=run_logger,
        )

    for source_client in clients:
        model = latest_client_models[source_client.name]
        for eval_client in clients:
            key = f"client_{source_client.name}_on_{eval_client.name}"
            final_evaluations[key] = _persist_model_evaluation(
                model=model,
                model_name="federated_dino",
                dataset_name=key,
                dataset_path=eval_client.path,
                splits=eval_client.splits,
                class_names=mapping.common_class_names,
                config=config,
                computer_configuration=computer_configuration,
                run_directory=run_directory,
                docs_root=docs_root,
                started_at=started_at,
                logger=run_logger,
            )

    torch.save(
        {"model_name": "federated_dino", "state_dict": global_model.state_dict()},
        run_directory / f"{run_directory.name}.pth",
    )
    for client_name, model in latest_client_models.items():
        torch.save(
            {
                "model_name": f"federated_dino_client_{client_name}",
                "state_dict": model.state_dict(),
            },
            run_directory / f"client_{client_name}.pth",
        )

    dump_metrics(
        run_directory,
        {
            "round_metrics": round_metrics,
            "final_evaluations": final_evaluations,
            "rounds": rounds,
            "local_epochs": local_epochs,
        },
    )
    _write_summary(run_directory, docs_root, "federated_dino", "sdss_decals")
    manifest = build_manifest(
        project_root=PROJECT_ROOT,
        dataset_path=clients[0].path,
        started_at=started_at,
        completed_at=now_iso(),
    )
    dump_manifest(run_directory, manifest)
    logger.info("federated run written to %s", run_directory)
    _close_logger(run_logger)
    return run_directory


def _load_dataset(
    dataset_name: str,
    label_mapping: dict[int, int],
    config: dict[str, Any],
    computer_configuration: dict[str, Any],
    raw_directory: Path,
    processed_directory: Path,
) -> LoadedDataset:
    training = config.get("training") or {}
    dataset_path = resolve_dataset_path(dataset_name, raw_directory, processed_directory)
    images, labels = read_dataset(dataset_path)
    images, labels = apply_label_mapping(images, labels, label_mapping)
    device = resolve_device(computer_configuration)
    splits = build_data_loaders(
        images=images,
        labels=labels,
        image_size=int(training.get("image_size", 224)),
        batch_size=int(training.get("batch_size", 32)),
        num_workers=resolve_worker_count(computer_configuration),
        random_seed=int(training.get("random_seed", config.get("random_seed", 42))),
        pin_memory=device.type == "cuda",
        split_ratios=config.get("split_ratios"),
    )
    return LoadedDataset(name=dataset_name, path=dataset_path, splits=splits)


def _train_local_model(
    *,
    adapter: DinoAdapter,
    model: torch.nn.Module,
    splits: Any,
    configuration: dict[str, Any],
    computer_configuration: dict[str, Any],
    device: torch.device,
    amp_enabled: bool,
    local_epochs: int,
    logger: logging.Logger,
) -> None:
    training = configuration.get("training") or {}
    criterion = adapter.build_criterion(splits, configuration, device)
    optimizer = adapter.build_optimizer(
        model,
        configuration,
        learning_rate=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    scaler = GradScaler(device.type, enabled=amp_enabled)
    for _ in range(local_epochs):
        adapter._run_epoch(
            model,
            splits.train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
        )


def _weighted_average_state_dicts(
    weighted_states: list[tuple[int, dict[str, torch.Tensor]]],
) -> dict[str, torch.Tensor]:
    total_weight = float(sum(weight for weight, _ in weighted_states))
    averaged: dict[str, torch.Tensor] = {}
    keys = weighted_states[0][1].keys()
    for key in keys:
        first_value = weighted_states[0][1][key]
        if not torch.is_floating_point(first_value):
            averaged[key] = first_value
            continue
        value = torch.zeros_like(first_value)
        for weight, state in weighted_states:
            value += state[key] * (float(weight) / total_weight)
        averaged[key] = value
    return averaged


def _persist_model_evaluation(
    *,
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    dataset_path: Path,
    splits: Any,
    class_names: list[str],
    config: dict[str, Any],
    computer_configuration: dict[str, Any],
    run_directory: Path,
    docs_root: Path,
    started_at: str,
    logger: logging.Logger,
) -> dict[str, Any]:
    device = resolve_device(computer_configuration)
    evaluation = _evaluate_torch_model(model, splits, computer_configuration, device, logger)
    metrics = _persist_evaluation(
        run_directory=run_directory,
        docs_root=docs_root,
        model_name=model_name,
        dataset_name=dataset_name,
        evaluation=evaluation,
        history=TrainingHistory(),
        splits=splits,
        class_names=class_names,
        config=config,
        dataset_path=dataset_path,
        started_at=started_at,
        is_deep_learning=True,
        logger=logger,
        append_to_metrics_file=False,
    )
    return metrics


def _evaluate_torch_model(
    model: torch.nn.Module,
    splits: Any,
    computer_configuration: dict[str, Any],
    device: torch.device,
    logger: logging.Logger,
) -> EvaluationResult:
    model.eval()
    amp_enabled = use_mixed_precision(computer_configuration, device)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    probability_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch_images, batch_labels in splits.test_loader:
            batch_images = batch_images.to(device, non_blocking=True)
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                logits = model(batch_images)
            probabilities = torch.softmax(logits.float(), dim=1)
            predictions.append(probabilities.argmax(dim=1).cpu().numpy())
            probability_chunks.append(probabilities.cpu().numpy())
            targets.append(batch_labels.numpy())
    all_predictions = np.concatenate(predictions)
    all_targets = np.concatenate(targets)
    all_probabilities = np.concatenate(probability_chunks, axis=0)
    accuracy = float((all_predictions == all_targets).mean())
    logger.info("evaluation accuracy=%.4f", accuracy)
    return EvaluationResult(
        accuracy=accuracy,
        predictions=all_predictions,
        targets=all_targets,
        classification_report=classification_report(
            all_targets, all_predictions, output_dict=True, zero_division=0
        ),
        confusion_matrix=compute_confusion_matrix(all_targets, all_predictions),
        probabilities=all_probabilities,
    )


def _persist_evaluation(
    *,
    run_directory: Path,
    docs_root: Path,
    model_name: str,
    dataset_name: str,
    evaluation: EvaluationResult,
    history: TrainingHistory,
    splits: Any,
    class_names: list[str],
    config: dict[str, Any],
    dataset_path: Path,
    started_at: str,
    is_deep_learning: bool,
    logger: logging.Logger,
    append_to_metrics_file: bool = True,
) -> dict[str, Any]:
    aggregate_metrics = compute_aggregate_metrics(
        targets=evaluation.targets,
        predictions=evaluation.predictions,
        probabilities=evaluation.probabilities,
        num_classes=splits.num_classes,
        is_deep_learning=is_deep_learning,
        top_k_values=_top_k_values(config),
    )
    error_analysis = compute_error_analysis(
        targets=evaluation.targets,
        predictions=evaluation.predictions,
        probabilities=evaluation.probabilities,
        num_classes=splits.num_classes,
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
    if append_to_metrics_file:
        dump_metrics(run_directory, metrics_payload)
    docs_directory = _write_summary(run_directory, docs_root, model_name, dataset_name)
    manifest = build_manifest(
        project_root=PROJECT_ROOT,
        dataset_path=dataset_path,
        started_at=started_at,
        completed_at=now_iso(),
    )
    dump_manifest(run_directory, manifest)
    jsonl_path = docs_root / (config.get("leaderboard") or {}).get(
        "jsonl_filename", "runs.jsonl"
    )
    append_run_to_jsonl(
        jsonl_path,
        build_runs_jsonl_record(
            run_directory=run_directory,
            documentation_directory=docs_directory,
            model_name=model_name,
            dataset_name=dataset_name,
            is_deep_learning=is_deep_learning,
            aggregate_metrics=aggregate_metrics,
            error_analysis=error_analysis,
            history_epochs=len(history.epochs),
            history_early_stopped=history.early_stopped,
            total_test_samples=int(len(evaluation.targets)),
            manifest=manifest,
        ),
    )
    logger.info(
        "%s on %s: accuracy=%.4f balanced_accuracy=%.4f",
        model_name,
        dataset_name,
        aggregate_metrics["accuracy"],
        aggregate_metrics["balanced_accuracy"],
    )
    return metrics_payload


def _write_summary(
    run_directory: Path,
    docs_root: Path,
    model_name: str,
    dataset_name: str,
) -> Path:
    docs_directory = docs_root / "models" / model_name / f"{run_directory.name}-{dataset_name}"
    docs_directory.mkdir(parents=True, exist_ok=True)
    summary_path = docs_directory / "summary.md"
    summary_path.write_text(
        "\n".join([
            f"# {model_name} — {dataset_name}",
            "",
            f"- Run directory: `{run_directory}`",
            "- Metrics are stored in `metrics.json` and `docs/runs.jsonl`.",
            "",
        ]),
        encoding="utf-8",
    )
    return docs_directory


def _dino_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    dino_config = (config.get("models") or {}).get("dino") or {}
    return {
        key: value
        for key, value in dino_config.items()
        if key != "fine_tuning"
    }


def _top_k_values(config: dict[str, Any]) -> list[int]:
    return list((config.get("evaluation") or {}).get("top_k_values") or [3, 5])


def _rebuild_leaderboard(config: dict[str, Any], docs_root: Path) -> None:
    leaderboard_section = config.get("leaderboard") or {}
    jsonl_path = docs_root / leaderboard_section.get("jsonl_filename", "runs.jsonl")
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


def _close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
