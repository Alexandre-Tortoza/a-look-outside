"""Inferência individual com DINO."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from modelos._avaliacao_padrao import inferir_modelo
from modelos.dino.modelo import DinoGalaxy


def inferir(
    imagem: np.ndarray,
    caminho_pesos: Path,
    num_classes: int = 10,
    tamanho_imagem: int = 224,
) -> tuple[int, float, np.ndarray]:
    return inferir_modelo(
        DinoGalaxy(modo="hub", backbone_hub="dinov2_vitb14"),
        imagem, caminho_pesos, num_classes, tamanho_imagem,
    )
