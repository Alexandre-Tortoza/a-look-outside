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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.analysis import generate_dataset_analysis  # noqa: E402
from dataset.balancing.registry import BalancingMethod, list_methods  # noqa: E402
from dataset.input_output import (  # noqa: E402
    DatasetEntry,
    class_distribution,
    derive_dataset_label,
    list_available_datasets,
    read_dataset,
    resolve_class_names,
    write_dataset,
)

console = Console()


def run(config: dict[str, Any]) -> None:
    paths = config.get("paths", {})
    raw_directory = PROJECT_ROOT / paths.get("raw_dataset_directory", "dataset/raw")
    processed_directory = PROJECT_ROOT / paths.get(
        "processed_dataset_directory", "dataset/processed"
    )

    available_datasets = list_available_datasets(raw_directory, processed_directory)
    if not available_datasets:
        console.print(
            "[yellow]Nenhum dataset .h5 encontrado em "
            f"{raw_directory} ou {processed_directory}.[/yellow]"
        )
        return

    action = inquirer.select(
        message="O que fazer?",
        choices=[
            Choice(value="balance", name="Balancear (e analisar) datasets"),
            Choice(value="analyze", name="Apenas analisar datasets"),
            Choice(value=None, name="Sair"),
        ],
        default="balance",
    ).execute()

    if action is None:
        return
    if action == "balance":
        _run_balance(config, available_datasets, processed_directory)
    else:
        _run_analyze(config, available_datasets)


def _run_balance(
    config: dict[str, Any],
    available_datasets: list[DatasetEntry],
    processed_directory: Path,
) -> None:
    random_seed = int(config.get("random_seed", 42))

    selected_datasets = _prompt_datasets(available_datasets)
    if not selected_datasets:
        console.print("[yellow]Nenhum dataset selecionado. Abortando.[/yellow]")
        return

    selected_methods = _prompt_methods()
    if not selected_methods:
        console.print("[yellow]Nenhum metodo selecionado. Abortando.[/yellow]")
        return

    plan_table = _build_plan_table(selected_datasets, selected_methods, processed_directory)
    console.print(plan_table)

    if not inquirer.confirm(message="Confirmar execucao?", default=True).execute():
        console.print("[yellow]Execucao cancelada.[/yellow]")
        return

    method_names = [method.name for method in selected_methods]
    for dataset_entry in selected_datasets:
        output_path = _output_path(dataset_entry, method_names, processed_directory)
        if output_path.exists():
            overwrite = inquirer.confirm(
                message=(
                    f"{output_path.relative_to(PROJECT_ROOT)} ja existe. Sobrescrever?"
                ),
                default=False,
            ).execute()
            if not overwrite:
                console.print(f"[yellow]Pulando {dataset_entry.name}.[/yellow]")
                continue

        console.rule(f"[bold]{dataset_entry.name}")
        console.print(f"Lendo [cyan]{dataset_entry.path}[/cyan]")
        images, labels = read_dataset(dataset_entry.path)
        console.print(
            f"  shape inicial: images={images.shape} dtype={images.dtype} | "
            f"distribuicao={class_distribution(labels)}"
        )

        _analyze_input_dataset(config, dataset_entry, images, labels)

        for method in selected_methods:
            console.print(f"Aplicando [magenta]{method.name}[/magenta]...")
            images, labels = method.apply(images, labels, random_seed)
            console.print(
                f"  apos {method.name}: images={images.shape} | "
                f"distribuicao={class_distribution(labels)}"
            )

        console.print(f"Escrevendo [green]{output_path}[/green]")
        write_dataset(output_path, images, labels)
        console.print("[green]ok[/green]")

        _analyze_output_dataset(
            config=config,
            output_path=output_path,
            images=images,
            labels=labels,
        )


def _run_analyze(
    config: dict[str, Any],
    available_datasets: list[DatasetEntry],
) -> None:
    selected_datasets = _prompt_datasets(available_datasets)
    if not selected_datasets:
        console.print("[yellow]Nenhum dataset selecionado. Abortando.[/yellow]")
        return

    docs_root = _docs_root(config)
    table = Table(title="Plano de analise")
    table.add_column("Dataset", style="cyan")
    table.add_column("Origem")
    table.add_column("Saida", style="green")
    for entry in selected_datasets:
        label = derive_dataset_label(entry)
        table.add_row(
            entry.name, entry.origin, str(docs_root.relative_to(PROJECT_ROOT) / label)
        )
    console.print(table)

    if not inquirer.confirm(message="Confirmar execucao?", default=True).execute():
        console.print("[yellow]Execucao cancelada.[/yellow]")
        return

    for entry in selected_datasets:
        console.rule(f"[bold]{entry.name}")
        images, labels = read_dataset(entry.path)
        output_directory = _run_dataset_analysis(
            config=config,
            dataset_label=derive_dataset_label(entry),
            images=images,
            labels=labels,
            source_path=entry.path,
        )
        console.print(
            f"  analise gravada em [green]{output_directory.relative_to(PROJECT_ROOT)}[/green]"
        )


def _analyze_input_dataset(
    config: dict[str, Any],
    dataset_entry: DatasetEntry,
    images: Any,
    labels: Any,
) -> None:
    output_directory = _run_dataset_analysis(
        config=config,
        dataset_label=derive_dataset_label(dataset_entry),
        images=images,
        labels=labels,
        source_path=dataset_entry.path,
    )
    console.print(
        f"  analise da entrada gravada em "
        f"[green]{output_directory.relative_to(PROJECT_ROOT)}[/green]"
    )


def _analyze_output_dataset(
    config: dict[str, Any],
    output_path: Path,
    images: Any,
    labels: Any,
) -> None:
    output_directory = _run_dataset_analysis(
        config=config,
        dataset_label=output_path.stem,
        images=images,
        labels=labels,
        source_path=output_path,
    )
    console.print(
        f"  analise da saida gravada em "
        f"[green]{output_directory.relative_to(PROJECT_ROOT)}[/green]"
    )


def _run_dataset_analysis(
    config: dict[str, Any],
    dataset_label: str,
    images: Any,
    labels: Any,
    source_path: Path,
) -> Path:
    docs_root = _docs_root(config)
    analysis_section = config.get("dataset_analysis") or {}
    sample_count_per_class = int(analysis_section.get("sample_count_per_class", 5))
    histogram_bin_count = int(
        analysis_section.get("intensity_histogram_bin_count", 64)
    )
    class_names = resolve_class_names(config.get("class_names"), dataset_label)
    random_seed = int(config.get("random_seed", 42))

    return generate_dataset_analysis(
        images=images,
        labels=labels,
        dataset_label=dataset_label,
        source_path=source_path,
        docs_root=docs_root,
        class_names=class_names,
        sample_count_per_class=sample_count_per_class,
        intensity_histogram_bin_count=histogram_bin_count,
        random_seed=random_seed,
    )


def _docs_root(config: dict[str, Any]) -> Path:
    analysis_section = config.get("dataset_analysis") or {}
    raw_value = analysis_section.get("output_directory", "docs/dataset")
    docs_root = PROJECT_ROOT / raw_value
    docs_root.mkdir(parents=True, exist_ok=True)
    return docs_root


def _prompt_datasets(datasets: list[DatasetEntry]) -> list[DatasetEntry]:
    choices = [
        Choice(value=i, name=entry.display_label) for i, entry in enumerate(datasets)
    ]
    selected_indices: list[int] = inquirer.checkbox(
        message="Selecione os datasets de entrada (espaco para marcar, enter para confirmar):",
        choices=choices,
        instruction="(use as setas e a barra de espaco)",
        validate=lambda result: len(result) > 0,
        invalid_message="Selecione ao menos um dataset.",
    ).execute()
    return [datasets[i] for i in selected_indices]


def _prompt_methods() -> list[BalancingMethod]:
    methods = list_methods()
    choices = [
        Choice(value=i, name=method.display_name) for i, method in enumerate(methods)
    ]
    console.print(
        "[dim]A ordem de selecao define a ordem de aplicacao dos metodos.[/dim]"
    )
    selected_indices: list[int] = inquirer.checkbox(
        message="Selecione os metodos de balanceamento (em ordem):",
        choices=choices,
        instruction="(use as setas e a barra de espaco)",
        validate=lambda result: len(result) > 0,
        invalid_message="Selecione ao menos um metodo.",
    ).execute()
    return [methods[i] for i in selected_indices]


def _build_plan_table(
    datasets: list[DatasetEntry],
    methods: list[BalancingMethod],
    processed_directory: Path,
) -> Table:
    table = Table(title="Plano de execucao")
    table.add_column("Dataset", style="cyan")
    table.add_column("Origem")
    table.add_column("Metodos", style="magenta")
    table.add_column("Saida", style="green")
    method_names = [method.name for method in methods]
    for entry in datasets:
        output_path = _output_path(entry, method_names, processed_directory)
        table.add_row(
            entry.name,
            entry.origin,
            " -> ".join(method_names),
            str(output_path.relative_to(PROJECT_ROOT)),
        )
    return table


def _output_path(
    dataset_entry: DatasetEntry,
    method_names: list[str],
    processed_directory: Path,
) -> Path:
    suffix = "_".join(method_names)
    output_name = f"{dataset_entry.name}_{suffix}.h5"
    return processed_directory / output_name


def _load_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


if __name__ == "__main__":
    run(_load_config())
