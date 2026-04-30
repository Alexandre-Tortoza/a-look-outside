"""Avaliação do ViT."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from modelos._avaliacao_padrao import avaliar_modelo
from modelos.vit.modelo import ViTGalaxy
from utils.config_loader import carregar_config, obter_config_modelo
from utils.metricas import ResultadoAvaliacao


def avaliar(caminho_pesos: Path, config_override: Optional[dict[str, Any]] = None) -> ResultadoAvaliacao:
    config = carregar_config()
    params = obter_config_modelo("vit", config)
    if config_override:
        params.update(config_override)
    return avaliar_modelo(
        ViTGalaxy(backbone=params.get("backbone", "vit_base_patch16_224"),
                  pretrained=params.get("pretrained", True)),
        caminho_pesos, params,
    )
