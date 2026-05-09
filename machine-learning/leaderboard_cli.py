from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import yaml
from InquirerPy import inquirer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_ROOT = Path(__file__).resolve().parent
for path in (str(PROJECT_ROOT), str(MODULE_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from leaderboard import (  # noqa: E402
    DEFAULT_HEATMAP_METRICS,
    load_records,
    regenerate_all,
)

console = Console()


def run(config: dict[str, Any] | None = None) -> None:
    if config is None:
        config = _load_config()

    paths = config.get("paths") or {}
    docs_root = PROJECT_ROOT / paths.get("documentation_directory", "docs")
    leaderboard_section = config.get("leaderboard") or {}
    jsonl_filename = leaderboard_section.get("jsonl_filename", "runs.jsonl")
    jsonl_path = docs_root / jsonl_filename

    if not jsonl_path.exists():
        console.print(
            f"[yellow]Nenhum runs.jsonl encontrado em "
            f"{jsonl_path.relative_to(PROJECT_ROOT)}.[/yellow] "
            "Rode pelo menos uma run em 'machine learning' antes."
        )
        return

    records = load_records(jsonl_path)
    if not records:
        console.print("[yellow]runs.jsonl esta vazio ou corrompido.[/yellow]")
        return

    console.print(
        f"[cyan]Encontradas {len(records)} runs em "
        f"{jsonl_path.relative_to(PROJECT_ROOT)}.[/cyan]"
    )

    if not inquirer.confirm(
        message="Regerar leaderboard a partir dessas runs?",
        default=True,
    ).execute():
        console.print("[yellow]Operacao cancelada.[/yellow]")
        return

    primary_metric = leaderboard_section.get("primary_metric", "balanced_accuracy")
    secondary_metric = leaderboard_section.get("secondary_metric", "accuracy")
    heatmap_metrics = list(
        leaderboard_section.get("heatmap_metrics") or DEFAULT_HEATMAP_METRICS
    )

    _build_logger()

    summary = regenerate_all(
        jsonl_path=jsonl_path,
        docs_root=docs_root,
        primary_metric=primary_metric,
        secondary_metric=secondary_metric,
        heatmap_metrics=heatmap_metrics,
    )

    _print_summary(summary, docs_root, jsonl_path, primary_metric)


def _print_summary(
    summary: dict[str, Any],
    docs_root: Path,
    jsonl_path: Path,
    primary_metric: str,
) -> None:
    table = Table(title="Leaderboard regenerado")
    table.add_column("Item", style="cyan")
    table.add_column("Valor")
    table.add_row("Total de runs", str(summary.get("records", 0)))
    table.add_row("(model, dataset) unicos", str(summary.get("best_records", 0)))
    table.add_row("Primary metric", primary_metric)
    table.add_row(
        "Leaderboard markdown",
        str(docs_root.relative_to(PROJECT_ROOT) / "leaderboard.md"),
    )
    table.add_row(
        "Leaderboard CSV",
        str(docs_root.relative_to(PROJECT_ROOT) / "leaderboard.csv"),
    )
    table.add_row(
        "JSONL fonte",
        str(jsonl_path.relative_to(PROJECT_ROOT)),
    )
    console.print(table)


def _build_logger() -> None:
    logger = logging.getLogger("leaderboard")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.addHandler(RichHandler(console=console, show_path=False, markup=True))


def _load_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


if __name__ == "__main__":
    run()
