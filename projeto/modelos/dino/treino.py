"""Ponto de entrada unificado para treino DINO.

Orquestra pré-treino (opcional) + fine-tuning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from modelos.dino import config as cfg
from modelos.dino.ajuste_fino import ajustar_fino
from modelos.dino.pre_treino import pre_treinar
from modelos.treinador import HistoricoTreino


def treinar(config_override: Optional[dict] = None) -> HistoricoTreino:
    """Treina o modelo DINO de ponta a ponta.

    Se MODO_DINO = "hub": pula pré-treino, faz apenas fine-tuning.
    Se MODO_DINO = "pre_treino": executa pré-treino + fine-tuning.
    """
    modo = (config_override or {}).get("modo", cfg.MODO_DINO)

    caminho_backbone: Optional[Path] = None
    if modo == "pre_treino":
        caminho_backbone = pre_treinar(config_override)

    return ajustar_fino(caminho_pesos_pretreino=caminho_backbone, config_override=config_override)


if __name__ == "__main__":
    print(treinar().resumo())
