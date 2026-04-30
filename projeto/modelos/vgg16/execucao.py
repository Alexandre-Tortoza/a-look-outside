"""Inferência individual com VGG16."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from modelos._avaliacao_padrao import inferir_modelo
from modelos.vgg16 import config as cfg
from modelos.vgg16.modelo import VGG16Galaxy


def inferir(
    imagem: np.ndarray,
    caminho_pesos: Path,
    num_classes: int = 10,
    tamanho_imagem: int = cfg.TAMANHO_IMAGEM,
) -> tuple[int, float, np.ndarray]:
    return inferir_modelo(VGG16Galaxy(pretrained=False), imagem, caminho_pesos,
                          num_classes, tamanho_imagem)
