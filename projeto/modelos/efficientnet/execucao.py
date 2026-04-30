"""Inferência individual com EfficientNet."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from modelos._avaliacao_padrao import inferir_modelo
from modelos.efficientnet import config as cfg
from modelos.efficientnet.modelo import EfficientNetGalaxy


def inferir(
    imagem: np.ndarray,
    caminho_pesos: Path,
    num_classes: int = 10,
    tamanho_imagem: int = cfg.TAMANHO_IMAGEM,
) -> tuple[int, float, np.ndarray]:
    return inferir_modelo(EfficientNetGalaxy(pretrained=False), imagem, caminho_pesos,
                          num_classes, tamanho_imagem)
