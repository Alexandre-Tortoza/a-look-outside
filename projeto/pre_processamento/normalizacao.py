"""Transforms de normalização e aumento de dados para uso em DataLoader."""

from __future__ import annotations

import numpy as np
from torchvision import transforms

from pre_processamento.config import (
    BRILHO_CONTRASTE_FATOR,
    DESVIO_IMAGENET,
    GRAU_ROTACAO_MAX,
    MEDIA_IMAGENET,
    PROB_FLIP_HORIZONTAL,
    PROB_FLIP_VERTICAL,
    PROB_ROTACAO,
    TAMANHO_PADRAO,
)

STATS_IMAGENET = (MEDIA_IMAGENET, DESVIO_IMAGENET)


def calcular_estatisticas(imagens: np.ndarray) -> tuple[list[float], list[float]]:
    """Calcula média e desvio padrão por canal a partir do conjunto de treino.

    Args:
        imagens: Array (N, H, W, C) ou (N, H, W) com valores uint8 ou float.

    Returns:
        Tupla (media_por_canal, desvio_por_canal) como listas de floats.
    """
    dados = imagens.astype(np.float32) / 255.0
    if dados.ndim == 3:
        dados = dados[..., np.newaxis]

    n_canais = dados.shape[-1]
    media = [float(dados[..., c].mean()) for c in range(n_canais)]
    desvio = [float(dados[..., c].std()) for c in range(n_canais)]
    return media, desvio


def obter_transform_treino(
    tamanho_imagem: int = TAMANHO_PADRAO,
    media: list[float] = MEDIA_IMAGENET,
    desvio: list[float] = DESVIO_IMAGENET,
    aumentar: bool = True,
) -> transforms.Compose:
    """Retorna pipeline de transforms para o conjunto de treino.

    Args:
        tamanho_imagem: Tamanho alvo após resize.
        media: Média por canal para normalização.
        desvio: Desvio padrão por canal para normalização.
        aumentar: Se False, omite transforms aleatórios (útil para reprodução).
    """
    passos: list = [
        transforms.ToPILImage(),
        transforms.Resize((tamanho_imagem, tamanho_imagem)),
    ]

    if aumentar:
        passos += [
            transforms.RandomHorizontalFlip(p=PROB_FLIP_HORIZONTAL),
            transforms.RandomVerticalFlip(p=PROB_FLIP_VERTICAL),
            transforms.RandomApply(
                [transforms.RandomRotation(degrees=GRAU_ROTACAO_MAX)],
                p=PROB_ROTACAO,
            ),
            transforms.ColorJitter(
                brightness=BRILHO_CONTRASTE_FATOR,
                contrast=BRILHO_CONTRASTE_FATOR,
            ),
        ]

    passos += [
        transforms.ToTensor(),
        transforms.Normalize(mean=media, std=desvio),
    ]
    return transforms.Compose(passos)


def obter_transform_avaliacao(
    tamanho_imagem: int = TAMANHO_PADRAO,
    media: list[float] = MEDIA_IMAGENET,
    desvio: list[float] = DESVIO_IMAGENET,
) -> transforms.Compose:
    """Retorna pipeline de transforms para validação e teste (sem augmentation)."""
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((tamanho_imagem, tamanho_imagem)),
            transforms.ToTensor(),
            transforms.Normalize(mean=media, std=desvio),
        ]
    )
