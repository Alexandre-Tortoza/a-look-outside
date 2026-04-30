"""Avaliação do DINO."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from modelos._avaliacao_padrao import avaliar_modelo
from modelos.dino.modelo import DinoGalaxy
from utils.config_loader import carregar_config, obter_config_modelo
from utils.metricas import ResultadoAvaliacao


def avaliar(caminho_pesos: Path, config_override: Optional[dict[str, Any]] = None) -> ResultadoAvaliacao:
    config = carregar_config()
    params = obter_config_modelo("dino", config)
    if config_override:
        params.update(config_override)
    return avaliar_modelo(
        DinoGalaxy(modo=params.get("modo", "hub"),
                   backbone_hub=params.get("backbone", "dinov2_vitb14"),
                   backbone_scratch=params.get("backbone_scratch", "vit_small_patch16_224")),
        caminho_pesos, params,
    )
