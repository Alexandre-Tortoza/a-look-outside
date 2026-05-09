from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import yaml
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MACHINE_LEARNING_ROOT = PROJECT_ROOT / "machine-learning"
for path in (str(PROJECT_ROOT), str(MACHINE_LEARNING_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from computer_configuration import (  # noqa: E402
    load_or_create_configuration,
)
from models.registry import get_model_info  # noqa: E402

from xai.artifact_storage import RunArtifact, discover_runs  # noqa: E402
from xai.explanation_generation import (  # noqa: E402
    ExplanationResult,
    generate_for_run,
)
from xai.methods.registry import XaiMethod, list_methods  # noqa: E402

console = Console()


def run(config: dict[str, Any]) -> None:
    paths = config.get("paths") or {}
    runs_root = PROJECT_ROOT / paths.get(
        "machine_learning_run_directory", "machine-learning/runs"
    )

    run_artifacts = discover_runs(runs_root)
    if not run_artifacts:
        console.print(
            "[yellow]Nenhuma run encontrada em "
            f"{runs_root.relative_to(PROJECT_ROOT)}.[/yellow] "
            "Treine um modelo antes via 'machine-learning'."
        )
        return

    selected_runs = _prompt_runs(run_artifacts)
    if not selected_runs:
        console.print("[yellow]Nenhuma run selecionada.[/yellow]")
        return

    methods = _prompt_methods(config)
    if not methods:
        console.print("[yellow]Nenhum metodo selecionado.[/yellow]")
        return

    computer_configuration_path = PROJECT_ROOT / paths.get(
        "machine_learning_computer_configuration_file",
        "machine-learning/my-computer.yaml",
    )
    computer_configuration, _ = load_or_create_configuration(computer_configuration_path)

    _print_plan(selected_runs, methods, paths)

    if not inquirer.confirm(message="Confirmar execucao?", default=True).execute():
        console.print("[yellow]Execucao cancelada.[/yellow]")
        return

    logger = _build_logger()
    results: list[ExplanationResult] = []
    for run_artifact in selected_runs:
        result = generate_for_run(
            run_artifact=run_artifact,
            methods=methods,
            config=config,
            computer_configuration=computer_configuration,
            logger=logger,
        )
        results.append(result)

    _print_results(results)


def _prompt_runs(run_artifacts: list[RunArtifact]) -> list[RunArtifact]:
    return inquirer.checkbox(
        message="Selecione as runs:",
        choices=[
            Choice(value=artifact, name=artifact.display_label)
            for artifact in run_artifacts
        ],
        validate=lambda result: len(result) > 0,
        invalid_message="Selecione ao menos uma run.",
    ).execute()


def _prompt_methods(config: dict[str, Any]) -> list[XaiMethod]:
    available_methods = list_methods()
    defaults = (config.get("xai") or {}).get("default_methods") or [
        method.name for method in available_methods
    ]
    choices = [
        Choice(
            value=method,
            name=method.display_name,
            enabled=method.name in defaults,
        )
        for method in available_methods
    ]
    return inquirer.checkbox(
        message="Selecione os metodos XAI (apenas os aplicaveis ao modelo de cada run rodam):",
        choices=choices,
        validate=lambda result: len(result) > 0,
        invalid_message="Selecione ao menos um metodo.",
    ).execute()


def _print_plan(
    selected_runs: list[RunArtifact],
    methods: list[XaiMethod],
    paths: dict[str, Any],
) -> None:
    xai_root = paths.get("xai_output_directory", "xai")
    table = Table(title="Plano de execucao XAI")
    table.add_column("Run", style="cyan")
    table.add_column("Modelo", style="magenta")
    table.add_column("Dataset", style="green")
    table.add_column("Metodos aplicaveis", style="yellow")
    table.add_column("Saida", style="dim")

    for artifact in selected_runs:
        info = get_model_info(artifact.model_name)
        applicable = [method.name for method in methods if method.applies_to(info)]
        if not applicable:
            applicable_text = "[nenhum aplicavel — sera ignorado]"
        else:
            applicable_text = ", ".join(applicable)
        output_path = f"{xai_root}/{artifact.model_name}/{artifact.dataset_name}"
        table.add_row(
            artifact.run_directory.name,
            artifact.model_name,
            artifact.dataset_name,
            applicable_text,
            output_path,
        )
    console.print(table)


def _print_results(results: list[ExplanationResult]) -> None:
    if not results:
        console.print("[yellow]Nenhum resultado.[/yellow]")
        return
    table = Table(title="Resultados XAI")
    table.add_column("Run", style="cyan")
    table.add_column("Samples", style="magenta")
    table.add_column("Artefatos por metodo", style="green")
    table.add_column("Saida", style="dim")
    for result in results:
        artefacts_text = ", ".join(
            f"{name}={count}" for name, count in result.method_artifact_counts.items()
        ) or "[nenhum metodo aplicavel]"
        relative_output = result.explanations_directory.parent.relative_to(PROJECT_ROOT)
        table.add_row(
            result.run_artifact.run_directory.name,
            str(result.sample_count),
            artefacts_text,
            str(relative_output),
        )
    console.print(table)


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("xai")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.addHandler(RichHandler(console=console, show_path=False, markup=True))
    return logger


def _load_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


if __name__ == "__main__":
    run(_load_config())
