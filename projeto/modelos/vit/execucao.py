"""Inferência individual com ViT."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from modelos._avaliacao_padrao import inferir_modelo
from modelos.vit import config as cfg
from modelos.vit.modelo import ViTGalaxy


def inferir(
    imagem: np.ndarray,
    caminho_pesos: Path,
    num_classes: int = 10,
    tamanho_imagem: int = cfg.TAMANHO_IMAGEM,
) -> tuple[int, float, np.ndarray]:
    return inferir_modelo(ViTGalaxy(pretrained=False), imagem, caminho_pesos,
                          num_classes, tamanho_imagem)
