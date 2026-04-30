"""Upscale offline de imagens pequenas para tamanho alvo (ex: 69px -> 224px)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from pre_processamento.config import TAMANHO_PADRAO


def aplicar_upscale(
    imagens: np.ndarray,
    tamanho_alvo: int = TAMANHO_PADRAO,
    metodo: int = Image.LANCZOS,
) -> np.ndarray:
    """Redimensiona todas as imagens para (tamanho_alvo, tamanho_alvo).

    Usa LANCZOS por padrão, que produz resultados melhores que bilinear
    para upscale de imagens pequenas (ex: SDSS 69px -> 224px).

    Args:
        imagens: Array (N, H, W, C) uint8.
        tamanho_alvo: Tamanho de saida (altura = largura).
        metodo: Metodo de interpolacao PIL (default: LANCZOS).

    Returns:
        Array (N, tamanho_alvo, tamanho_alvo, C) uint8.
    """
    if imagens.shape[1] == tamanho_alvo and imagens.shape[2] == tamanho_alvo:
        return imagens

    n, _, _, c = imagens.shape
    saida = np.empty((n, tamanho_alvo, tamanho_alvo, c), dtype=imagens.dtype)

    for i in range(n):
        img = Image.fromarray(imagens[i])
        img = img.resize((tamanho_alvo, tamanho_alvo), metodo)
        saida[i] = np.array(img)

    return saida
