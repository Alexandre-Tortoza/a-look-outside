"""Carregamento e merge de configuracao centralizada (config.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


_RAIZ_PROJETO = Path(__file__).resolve().parent.parent
_CAMINHO_PADRAO = _RAIZ_PROJETO / "config.yaml"


def carregar_config(caminho: str | Path | None = None) -> dict[str, Any]:
    """Carrega o config.yaml e retorna o dict completo.

    Args:
        caminho: Caminho do YAML. Se None, usa ``projeto/config.yaml``.
    """
    caminho = Path(caminho) if caminho else _CAMINHO_PADRAO
    with open(caminho, encoding="utf-8") as f:
        return yaml.safe_load(f)


def obter_config_modelo(nome_modelo: str, config: dict[str, Any]) -> dict[str, Any]:
    """Retorna config merged: global + modelo especifico.

    Campos do modelo sobrescrevem os do global.

    Args:
        nome_modelo: Chave em ``config["modelos"]`` (ex: ``"cnn"``, ``"resnet50"``).
        config: Dict completo retornado por ``carregar_config()``.

    Returns:
        Dict com todos os hiperparametros prontos para uso.
    """
    cfg_global = dict(config.get("global", {}))
    cfg_modelo = dict(config.get("modelos", {}).get(nome_modelo, {}))
    merged = {**cfg_global, **cfg_modelo}
    merged["nome_modelo"] = nome_modelo
    return merged


def obter_config_recursos(config: dict[str, Any]) -> dict[str, Any]:
    """Retorna a secao de recursos de hardware."""
    return dict(config.get("recursos", {}))


def obter_experimento(nome: str, config: dict[str, Any]) -> dict[str, Any]:
    """Retorna a config de um experimento pre-definido.

    Raises:
        KeyError: se o experimento nao existir.
    """
    exps = config.get("experimentos", {})
    if nome not in exps:
        disponiveis = ", ".join(exps.keys())
        raise KeyError(
            f"Experimento '{nome}' nao encontrado. Disponiveis: {disponiveis}"
        )
    return dict(exps[nome])


def listar_experimentos(config: dict[str, Any]) -> list[str]:
    """Lista os nomes de todos os experimentos definidos no YAML."""
    return list(config.get("experimentos", {}).keys())


def listar_modelos(config: dict[str, Any]) -> list[str]:
    """Lista os nomes de todos os modelos definidos no YAML."""
    return list(config.get("modelos", {}).keys())
