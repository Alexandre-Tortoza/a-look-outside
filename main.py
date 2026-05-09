from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from rich.console import Console

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

console = Console()


MODULE_OPTIONS: list[tuple[str, str, str]] = [
    ("benchmark", "benchmark.main", "Rodar pipeline declarativo completo (dataset->ml->xai)"),
    ("dataset balancing", "dataset.main", "Aplicar metodos de balanceamento aos datasets"),
    ("machine learning", "machine_learning_main", "Treinar e avaliar modelos"),
    ("xai", "xai.main", "Gerar explicacoes visuais"),
    ("regenerate leaderboard", "leaderboard_cli", "Regerar leaderboard a partir de runs.jsonl"),
]


def main() -> None:
    config = _load_config()
    while True:
        choice = inquirer.select(
            message="A Look Outside - selecione um modulo:",
            choices=[
                *(
                    Choice(value=module_path, name=f"{label} - {description}")
                    for label, module_path, description in MODULE_OPTIONS
                ),
                Choice(value=None, name="sair"),
            ],
            default=MODULE_OPTIONS[0][1],
        ).execute()

        if choice is None:
            console.print("[dim]Ate mais.[/dim]")
            return

        run_callable = _resolve_run_callable(choice)
        try:
            run_callable(config)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrompido pelo usuario.[/yellow]")
        except Exception as error:  # noqa: BLE001
            console.print(f"[red]Erro ao executar {choice}:[/red] {error}")


def _resolve_run_callable(module_path: str) -> Callable[[dict[str, Any]], None]:
    if module_path == "machine_learning_main":
        module_file = PROJECT_ROOT / "machine-learning" / "main.py"
        return _load_run_from_file(module_file, module_alias="machine_learning_main")
    if module_path == "leaderboard_cli":
        module_file = PROJECT_ROOT / "machine-learning" / "leaderboard_cli.py"
        return _load_run_from_file(module_file, module_alias="leaderboard_cli")
    module = importlib.import_module(module_path)
    return module.run


def _load_run_from_file(module_file: Path, module_alias: str) -> Callable[[dict[str, Any]], None]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_alias, module_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module from {module_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_alias] = module
    spec.loader.exec_module(module)
    return module.run


def _load_config() -> dict[str, Any]:
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


if __name__ == "__main__":
    main()
