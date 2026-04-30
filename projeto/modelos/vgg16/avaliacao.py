"""Avaliação do VGG16."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from modelos._avaliacao_padrao import avaliar_modelo
from modelos.vgg16.modelo import VGG16Galaxy
from utils.config_loader import carregar_config, obter_config_modelo
from utils.metricas import ResultadoAvaliacao


def avaliar(caminho_pesos: Path, config_override: Optional[dict[str, Any]] = None) -> ResultadoAvaliacao:
    config = carregar_config()
    params = obter_config_modelo("vgg16", config)
    if config_override:
        params.update(config_override)
    return avaliar_modelo(
        VGG16Galaxy(pretrained=params.get("pretrained", True)),
        caminho_pesos, params,
    )
