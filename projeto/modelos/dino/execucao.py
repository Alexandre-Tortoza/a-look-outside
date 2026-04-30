"""Inferência individual com DINO."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from modelos._avaliacao_padrao import inferir_modelo
from modelos.dino import config as cfg
from modelos.dino.modelo import DinoGalaxy


def inferir(
    imagem: np.ndarray,
    caminho_pesos: Path,
    num_classes: int = 10,
    tamanho_imagem: int = cfg.TAMANHO_IMAGEM,
) -> tuple[int, float, np.ndarray]:
    return inferir_modelo(
        DinoGalaxy(modo=cfg.MODO_DINO, backbone_hub=cfg.BACKBONE_DINO),
        imagem, caminho_pesos, num_classes, tamanho_imagem,
    )
