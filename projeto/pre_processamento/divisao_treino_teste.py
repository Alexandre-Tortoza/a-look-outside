"""Divisão estratificada de datasets em treino, validação e teste."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from pre_processamento.config import FRACAO_TESTE, FRACAO_TREINO, FRACAO_VAL


@dataclass
class DivisaoDados:
    """Armazena as três partições do dataset com imagens e rótulos."""

    imagens_treino: np.ndarray
    rotulos_treino: np.ndarray
    imagens_val: np.ndarray
    rotulos_val: np.ndarray
    imagens_teste: np.ndarray
    rotulos_teste: np.ndarray

    @property
    def n_treino(self) -> int:
        return len(self.rotulos_treino)

    @property
    def n_val(self) -> int:
        return len(self.rotulos_val)

    @property
    def n_teste(self) -> int:
        return len(self.rotulos_teste)

    def __repr__(self) -> str:
        return (
            f"DivisaoDados(treino={self.n_treino}, val={self.n_val}, teste={self.n_teste})"
        )


def dividir_estratificado(
    imagens: np.ndarray,
    rotulos: np.ndarray,
    fracao_treino: float = FRACAO_TREINO,
    fracao_val: float = FRACAO_VAL,
    semente: int = 42,
) -> DivisaoDados:
    """Divide o dataset de forma estratificada preservando proporção de classes.

    Args:
        imagens: Array (N, H, W, C).
        rotulos: Array (N,) de inteiros.
        fracao_treino: Proporção do conjunto de treino.
        fracao_val: Proporção da validação (o restante vira teste).
        semente: Semente para reprodutibilidade.

    Returns:
        DivisaoDados com as três partições.
    """
    fracao_teste = 1.0 - fracao_treino - fracao_val
    assert fracao_teste > 0, "A soma de fracao_treino + fracao_val deve ser < 1."

    # Primeiro split: treino vs (val + teste)
    imgs_treino, imgs_restante, rot_treino, rot_restante = train_test_split(
        imagens,
        rotulos,
        train_size=fracao_treino,
        stratify=rotulos,
        random_state=semente,
    )

    # Segundo split: val vs teste (proporcional ao restante)
    val_sobre_restante = fracao_val / (fracao_val + fracao_teste)
    imgs_val, imgs_teste, rot_val, rot_teste = train_test_split(
        imgs_restante,
        rot_restante,
        train_size=val_sobre_restante,
        stratify=rot_restante,
        random_state=semente,
    )

    return DivisaoDados(
        imagens_treino=imgs_treino,
        rotulos_treino=rot_treino,
        imagens_val=imgs_val,
        rotulos_val=rot_val,
        imagens_teste=imgs_teste,
        rotulos_teste=rot_teste,
    )
