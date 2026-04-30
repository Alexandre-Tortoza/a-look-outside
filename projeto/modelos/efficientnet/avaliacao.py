"""Avaliação do EfficientNet."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from modelos._avaliacao_padrao import avaliar_modelo
from modelos.efficientnet.modelo import EfficientNetGalaxy
from utils.config_loader import carregar_config, obter_config_modelo
from utils.metricas import ResultadoAvaliacao


def avaliar(caminho_pesos: Path, config_override: Optional[dict[str, Any]] = None) -> ResultadoAvaliacao:
    config = carregar_config()
    params = obter_config_modelo("efficientnet", config)
    if config_override:
        params.update(config_override)
    return avaliar_modelo(
        EfficientNetGalaxy(params.get("variante_backbone", "efficientnet_b0"),
                           params.get("pretrained", True)),
        caminho_pesos, params,
    )
