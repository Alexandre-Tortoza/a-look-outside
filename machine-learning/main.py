from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_ROOT = Path(__file__).resolve().parent
for path in (str(PROJECT_ROOT), str(MODULE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from computer_configuration import (  # noqa: E402
    load_or_create_configuration,
    resolve_device,
)
from models.registry import list_models  # noqa: E402
from pipeline import ModelSpec, RunResult, run_pipeline  # noqa: E402

from dataset.input_output import list_available_datasets  # noqa: E402

console = Console()


def run(config: dict[str, Any]) -> None:
    paths = config.get("paths", {})
    computer_configuration_path = PROJECT_ROOT / paths.get(
        "machine_learning_computer_configuration_file",
        "machine-learning/my-computer.yaml",
    )

    computer_configuration, was_created = load_or_create_configuration(
        computer_configuration_path
    )
    if was_created:
        console.print(
            "[green]my-computer.yaml gerado em "
            f"{computer_configuration_path.relative_to(PROJECT_ROOT)}[/green]"
        )
        _print_computer_summary(computer_configuration)
        confirm_message = (
            "Continuar com estes recursos? "
            "(escolha 'nao' para editar o arquivo manualmente antes)"
        )
        if not inquirer.confirm(message=confirm_message, default=True).execute():
            console.print(
                "[yellow]Edite o arquivo e rode novamente.[/yellow]"
            )
            return

    mode = inquirer.select(
        message="Como executar?",
        choices=[
            Choice(value="manual", name="Selecao manual de modelos e datasets"),
            Choice(value="pipeline", name="Pipeline pre-definido em config.yaml"),
        ],
        default="manual",
    ).execute()

    if mode == "manual":
        plan = _prompt_manual(config)
    else:
        plan = _prompt_pipeline(config)

    if plan is None:
        return

    model_specs, dataset_names = plan
    _print_plan(model_specs, dataset_names, computer_configuration)

    if not inquirer.confirm(message="Confirmar execucao?", default=True).execute():
        console.print("[yellow]Execucao cancelada.[/yellow]")
        return

    results = run_pipeline(
        model_specs=model_specs,
        dataset_names=dataset_names,
        configuration=config,
        computer_configuration=computer_configuration,
    )
    _print_results(results)


def _prompt_manual(
    config: dict[str, Any],
) -> tuple[list[ModelSpec], list[str]] | None:
    paths = config.get("paths", {})
    raw_directory = PROJECT_ROOT / paths.get("raw_dataset_directory", "dataset/raw")
    processed_directory = PROJECT_ROOT / paths.get(
        "processed_dataset_directory", "dataset/processed"
    )

    available_datasets = list_available_datasets(raw_directory, processed_directory)
    if not available_datasets:
        console.print(
            "[yellow]Nenhum dataset encontrado em "
            f"{raw_directory} ou {processed_directory}.[/yellow]"
        )
        return None

    selected_models: list[str] = inquirer.checkbox(
        message="Selecione os modelos:",
        choices=[
            Choice(value=info.name, name=info.display_name) for info in list_models()
        ],
        validate=lambda result: len(result) > 0,
        invalid_message="Selecione ao menos um modelo.",
    ).execute()

    model_specs: list[ModelSpec] = []
    knn_defaults = (config.get("models", {}) or {}).get("k_nearest_neighbors", {}) or {}
    for model_name in selected_models:
        kwargs: dict[str, Any] = {}
        if model_name == "k_nearest_neighbors":
            mode = inquirer.select(
                message="KNN: usar pixels achatados ou features de CNN pretrained?",
                choices=[
                    Choice(value="pixels", name="Pixels achatados"),
                    Choice(value="features", name="Features de ResNet50 ImageNet (congelada)"),
                ],
                default=knn_defaults.get("mode", "pixels"),
            ).execute()
            kwargs["mode"] = mode
            kwargs["n_neighbors"] = int(knn_defaults.get("n_neighbors", 5))
        model_specs.append(ModelSpec(name=model_name, factory_kwargs=kwargs))

    dataset_choices: list[Choice] = []
    for entry in available_datasets:
        dataset_id = (
            f"{entry.name}_raw" if entry.origin == "raw" else entry.name
        )
        dataset_choices.append(Choice(value=dataset_id, name=entry.display_label))

    selected_datasets: list[str] = inquirer.checkbox(
        message="Selecione os datasets:",
        choices=dataset_choices,
        validate=lambda result: len(result) > 0,
        invalid_message="Selecione ao menos um dataset.",
    ).execute()

    return model_specs, selected_datasets


def _prompt_pipeline(
    config: dict[str, Any],
) -> tuple[list[ModelSpec], list[str]] | None:
    pipelines = config.get("pipelines") or {}
    if not pipelines:
        console.print(
            "[yellow]Nenhum pipeline definido em config.yaml. "
            "Use o modo manual ou configure 'pipelines:'.[/yellow]"
        )
        return None

    pipeline_name = inquirer.select(
        message="Selecione o pipeline:",
        choices=[Choice(value=name, name=name) for name in pipelines],
    ).execute()
    spec = pipelines[pipeline_name]
    knn_defaults = (config.get("models", {}) or {}).get("k_nearest_neighbors", {}) or {}

    model_specs: list[ModelSpec] = []
    for model_name in spec.get("model_names", []):
        kwargs: dict[str, Any] = {}
        if model_name == "k_nearest_neighbors":
            kwargs["mode"] = knn_defaults.get("mode", "pixels")
            kwargs["n_neighbors"] = int(knn_defaults.get("n_neighbors", 5))
        model_specs.append(ModelSpec(name=model_name, factory_kwargs=kwargs))

    dataset_names = list(spec.get("dataset_names", []))
    if not model_specs or not dataset_names:
        console.print(
            f"[yellow]Pipeline '{pipeline_name}' nao define modelos ou datasets.[/yellow]"
        )
        return None
    return model_specs, dataset_names


def _print_computer_summary(configuration: dict[str, Any]) -> None:
    computer = configuration.get("computer", {})
    table = Table(title="Hardware detectado")
    table.add_column("Campo", style="cyan")
    table.add_column("Valor")
    for key, value in computer.items():
        table.add_row(key, str(value))
    console.print(table)


def _print_plan(
    model_specs: list[ModelSpec],
    dataset_names: list[str],
    computer_configuration: dict[str, Any],
) -> None:
    device = resolve_device(computer_configuration)
    table = Table(title=f"Plano de execucao (device={device})")
    table.add_column("Modelo", style="cyan")
    table.add_column("Variante", style="magenta")
    table.add_column("Datasets", style="green")
    for spec in model_specs:
        variant = (
            spec.factory_kwargs.get("mode", "-")
            if spec.name == "k_nearest_neighbors"
            else "-"
        )
        table.add_row(spec.name, variant, ", ".join(dataset_names))
    console.print(table)


def _print_results(results: list[RunResult]) -> None:
    if not results:
        console.print("[yellow]Nenhum resultado.[/yellow]")
        return
    table = Table(title="Resultados")
    table.add_column("Modelo", style="cyan")
    table.add_column("Dataset", style="green")
    table.add_column("Accuracy", style="magenta")
    table.add_column("Run", style="dim")
    table.add_column("Docs", style="dim")
    for result in results:
        table.add_row(
            result.model_name,
            result.dataset_name,
            f"{result.evaluation.accuracy:.4f}",
            str(result.run_directory.relative_to(PROJECT_ROOT)),
            str(result.documentation_directory.relative_to(PROJECT_ROOT)),
        )
    console.print(table)


def _load_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


if __name__ == "__main__":
    run(_load_config())
