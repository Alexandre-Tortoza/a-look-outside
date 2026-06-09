from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from rich.console import Console
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MACHINE_LEARNING_ROOT = PROJECT_ROOT / "machine-learning"
for path in (str(PROJECT_ROOT), str(MACHINE_LEARNING_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from computer_configuration import (  # noqa: E402
    load_or_create_configuration,
)

from benchmark.orchestrator import (  # noqa: E402
    BenchmarkResult,
    run_benchmark,
)

console = Console()


def run(
    config: dict[str, Any],
    benchmark_name: str | None = None,
    assume_yes: bool = False,
) -> None:
    benchmarks = config.get("benchmarks") or {}
    if not benchmarks:
        console.print(
            "[yellow]Nenhum benchmark definido em config.yaml. "
            "Adicione um bloco 'benchmarks:' antes de continuar.[/yellow]"
        )
        return

    if benchmark_name is None:
        benchmark_name = inquirer.select(
            message="Selecione o benchmark:",
            choices=[
                Choice(
                    value=name,
                    name=f"{name} — {(spec or {}).get('description', '')}",
                )
                for name, spec in benchmarks.items()
            ],
        ).execute()
    elif benchmark_name not in benchmarks:
        available = ", ".join(sorted(benchmarks))
        raise KeyError(f"benchmark '{benchmark_name}' not found. available: {available}")

    benchmark_spec = benchmarks[benchmark_name]
    _print_plan(benchmark_name, benchmark_spec, config)

    if not assume_yes and not inquirer.confirm(
        message="Confirmar execucao?", default=True
    ).execute():
        console.print("[yellow]Execucao cancelada.[/yellow]")
        return

    paths = config.get("paths") or {}
    computer_configuration_path = PROJECT_ROOT / paths.get(
        "machine_learning_computer_configuration_file",
        "machine-learning/my-computer.yaml",
    )
    computer_configuration, _ = load_or_create_configuration(
        computer_configuration_path
    )

    result = run_benchmark(
        benchmark_name=benchmark_name,
        config=config,
        computer_configuration=computer_configuration,
        project_root=PROJECT_ROOT,
        console=console,
    )

    _print_result(result, config)


def _print_plan(
    benchmark_name: str,
    benchmark_spec: dict[str, Any],
    config: dict[str, Any],
) -> None:
    if benchmark_spec.get("experiment") == "sequence":
        summary_table = Table(title=f"Plano de execucao — {benchmark_name}")
        summary_table.add_column("Item", style="cyan")
        summary_table.add_column("Valor")
        summary_table.add_row("Experiment", "sequence")
        summary_table.add_row("Steps", " -> ".join(benchmark_spec.get("steps") or []))
        console.print(summary_table)
        return

    if benchmark_spec.get("experiment") == "cross_dataset_federated":
        summary_table = Table(title=f"Plano de execucao — {benchmark_name}")
        summary_table.add_column("Item", style="cyan")
        summary_table.add_column("Valor")
        summary_table.add_row("Experiment", "cross_dataset_federated")
        summary_table.add_row(
            "Datasets", ", ".join(benchmark_spec.get("datasets") or [])
        )
        summary_table.add_row(
            "Top model count", str(benchmark_spec.get("top_model_count", 3))
        )
        federated = benchmark_spec.get("federated") or {}
        summary_table.add_row("FedAvg rounds", str(federated.get("rounds", 3)))
        summary_table.add_row(
            "Local epochs", str(federated.get("local_epochs", 1))
        )
        console.print(summary_table)
        return

    datasets_spec = benchmark_spec.get("datasets") or {}
    sources = list(datasets_spec.get("sources") or [])
    include_raw = bool(datasets_spec.get("include_raw", True))
    variants = list(datasets_spec.get("balancing_variants") or [])
    models = list(benchmark_spec.get("models") or [])
    knn_modes = list(benchmark_spec.get("knn_modes") or ["pixels"])

    dataset_count = len(sources) * (
        (1 if include_raw else 0) + len(variants)
    )
    knn_specs = (
        len(knn_modes) if "k_nearest_neighbors" in models else 0
    )
    other_models = [m for m in models if m != "k_nearest_neighbors"]
    estimated_runs = dataset_count * (len(other_models) + knn_specs)

    summary_table = Table(title=f"Plano de execucao — {benchmark_name}")
    summary_table.add_column("Item", style="cyan")
    summary_table.add_column("Valor")
    if benchmark_spec.get("protocol"):
        summary_table.add_row("Protocol", str(benchmark_spec.get("protocol")))
    if benchmark_spec.get("evidence_role"):
        summary_table.add_row("Evidence role", str(benchmark_spec.get("evidence_role")))
    summary_table.add_row("Sources", ", ".join(sources) or "—")
    summary_table.add_row("Include raw", "yes" if include_raw else "no")
    summary_table.add_row(
        "Balancing variants",
        ", ".join(" -> ".join(v.get("methods", [])) for v in variants) or "—",
    )
    summary_table.add_row("Models", ", ".join(models) or "—")
    summary_table.add_row(
        "KNN modes",
        ", ".join(knn_modes) if "k_nearest_neighbors" in models else "—",
    )
    summary_table.add_row("Datasets totais", str(dataset_count))
    summary_table.add_row("Runs estimadas", str(estimated_runs))
    summary_table.add_row(
        "Dataset analysis",
        "yes" if benchmark_spec.get("generate_dataset_analysis", True) else "no",
    )
    summary_table.add_row(
        "XAI", "yes" if benchmark_spec.get("generate_xai", False) else "no",
    )
    summary_table.add_row(
        "Recommendations",
        "yes" if benchmark_spec.get("generate_recommendations", True) else "no",
    )
    console.print(summary_table)


def _print_result(result: BenchmarkResult, config: dict[str, Any]) -> None:
    paths = config.get("paths") or {}
    docs_root = PROJECT_ROOT / paths.get("documentation_directory", "docs")

    summary_table = Table(title=f"Benchmark '{result.benchmark_name}' — resumo")
    summary_table.add_column("Item", style="cyan")
    summary_table.add_column("Valor")
    summary_table.add_row("Datasets resolvidos", str(len(result.resolved_datasets)))
    summary_table.add_row("Runs executadas", str(len(result.run_results)))
    summary_table.add_row("Insights gerados", str(len(result.insights)))
    for stage_name, duration in result.stage_durations_seconds.items():
        summary_table.add_row(f"Tempo {stage_name}", f"{duration:.1f}s")
    summary_table.add_row(
        "Leaderboard",
        str((docs_root / "leaderboard.md").relative_to(PROJECT_ROOT)),
    )
    summary_table.add_row(
        "Recommendations",
        str((docs_root / "recommendations.md").relative_to(PROJECT_ROOT)),
    )
    console.print(summary_table)

    top_results = sorted(
        result.run_results,
        key=lambda r: r.evaluation.accuracy,
        reverse=True,
    )[:3]
    if top_results:
        top_table = Table(title="Top 3 runs (by accuracy)")
        top_table.add_column("Modelo", style="cyan")
        top_table.add_column("Dataset", style="green")
        top_table.add_column("Accuracy", style="magenta")
        top_table.add_column("Run dir", style="dim")
        for run_result in top_results:
            top_table.add_row(
                run_result.model_name,
                run_result.dataset_name,
                f"{run_result.evaluation.accuracy:.4f}",
                str(run_result.run_directory.relative_to(PROJECT_ROOT)),
            )
        console.print(top_table)


def _load_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run declarative benchmarks.")
    parser.add_argument(
        "--benchmark",
        "-b",
        help="Benchmark name from config.yaml, e.g. everything.",
    )
    parser.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help="Run without the interactive confirmation prompt.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(_load_config(), benchmark_name=args.benchmark, assume_yes=args.yes)
