"""Ponto de entrada unificado para treino DINO.

Orquestra pré-treino (opcional) + fine-tuning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from modelos.dino.ajuste_fino import ajustar_fino
from modelos.dino.pre_treino import pre_treinar
from modelos.treinador import HistoricoTreino
from utils.config_loader import carregar_config, obter_config_modelo


def treinar(config_override: Optional[dict[str, Any]] = None) -> HistoricoTreino:
    """Treina o modelo DINO de ponta a ponta.

    Se modo = "hub": pula pré-treino, faz apenas fine-tuning.
    Se modo = "pre_treino": executa pré-treino + fine-tuning.
    """
    config = carregar_config()
    params = obter_config_modelo("dino", config)
    if config_override:
        params.update(config_override)

    modo = params.get("modo", "hub")

    caminho_backbone: Optional[Path] = None
    if modo == "pre_treino":
        caminho_backbone = pre_treinar(config_override)

    return ajustar_fino(caminho_pesos_pretreino=caminho_backbone, config_override=config_override)


if __name__ == "__main__":
    print(treinar().resumo())
