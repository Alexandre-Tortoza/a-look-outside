"""Geracao de nomes unicos de experimento e registro de historico de runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def gerar_nome_experimento(
    modelo: str,
    dataset: str,
    versao: str,
    epocas: int,
) -> str:
    """Gera nome unico de experimento com timestamp.

    Formato: ``{modelo}_{dataset}_{versao}_ep{epocas}_{YYYYMMDD_HHMMSS}``

    Exemplo: ``cnn_decals_raw_ep50_20260430_143022``
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{modelo}_{dataset}_{versao}_ep{epocas}_{timestamp}"


def registrar_run(
    dir_pesos: Path,
    nome_modelo: str,
    nome_experimento: str,
    params: dict[str, Any],
    historico: Optional[Any] = None,
) -> None:
    """Registra metadata de um run no historico JSONL do modelo.

    Cada linha eh um JSON com dados de um run, append-only.

    Arquivo: ``{dir_pesos}/{nome_modelo}/historico_runs.jsonl``
    """
    pasta = Path(dir_pesos) / nome_modelo
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / "historico_runs.jsonl"

    registro: dict[str, Any] = {
        "nome_experimento": nome_experimento,
        "timestamp": datetime.now().isoformat(),
        "hiperparametros": _limpar_para_json(params),
    }

    if historico is not None:
        registro["melhor_val_acc"] = getattr(historico, "melhor_val_acc", None)
        registro["epocas_totais"] = getattr(historico, "epocas_totais", None)
        registro["parou_cedo"] = getattr(historico, "parou_cedo", None)

    with open(caminho, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def carregar_historico(dir_pesos: Path, nome_modelo: str) -> list[dict[str, Any]]:
    """Carrega o historico de runs de um modelo.

    Returns:
        Lista de dicts, um por run, em ordem cronologica.
    """
    caminho = Path(dir_pesos) / nome_modelo / "historico_runs.jsonl"
    if not caminho.exists():
        return []

    runs = []
    for linha in caminho.read_text(encoding="utf-8").strip().splitlines():
        if linha:
            runs.append(json.loads(linha))
    return runs


def carregar_historico_todos(dir_pesos: Path) -> dict[str, list[dict[str, Any]]]:
    """Carrega historico de todos os modelos.

    Returns:
        Dict modelo -> lista de runs.
    """
    dir_pesos = Path(dir_pesos)
    todos: dict[str, list[dict[str, Any]]] = {}
    if not dir_pesos.exists():
        return todos
    for pasta_modelo in sorted(dir_pesos.iterdir()):
        if pasta_modelo.is_dir():
            runs = carregar_historico(dir_pesos, pasta_modelo.name)
            if runs:
                todos[pasta_modelo.name] = runs
    return todos


def _limpar_para_json(d: dict) -> dict:
    """Remove valores nao serializaveis."""
    limpo = {}
    for k, v in d.items():
        if isinstance(v, dict):
            limpo[k] = _limpar_para_json(v)
        elif isinstance(v, (str, int, float, bool, type(None))):
            limpo[k] = v
        elif isinstance(v, (list, tuple)):
            limpo[k] = list(v)
        else:
            limpo[k] = str(v)
    return limpo
