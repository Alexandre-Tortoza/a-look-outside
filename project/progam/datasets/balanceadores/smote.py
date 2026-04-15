"""
Balanceador SMOTE (Synthetic Minority Over-sampling Technique).

Gera amostras sintéticas da classe minoritária usando interpolação k-NN.
"""

from typing import Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .base import BalanceadorBase


class BalanceadorSMOTE(BalanceadorBase):
    """Implementação de SMOTE para balanceamento de datasets."""

    def __init__(
        self,
        n_vizinhos: int = 5,
        n_trabalhos: int = -1,
        semente: Optional[int] = None,
    ):
        """
        Inicializar balanceador SMOTE.

        Args:
            n_vizinhos: Número de vizinhos para k-NN.
            n_trabalhos: Número de jobs paralelos (-1 para usar todos).
            semente: Seed para reprodutibilidade.
        """
        super().__init__(semente=semente)
        self.n_vizinhos = n_vizinhos
        self.n_trabalhos = n_trabalhos

    def balancear(
        self,
        imagens: np.ndarray,
        rótulos: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplicar SMOTE.

        Args:
            imagens: Array com imagens [n_amostras, altura, largura, canais].
            rótulos: Array com rótulos [n_amostras].

        Returns:
            Tupla (imagens_balanceadas, rótulos_balanceados).
        """
        self.logger.info("Aplicando SMOTE...")

        # Remodelar imagens para 2D
        n_amostras, altura, largura, canais = imagens.shape
        imagens_2d = imagens.reshape(n_amostras, -1)

        # Obter distribuição original
        classes_unicas, contagens = np.unique(rótulos, return_counts=True)
        classe_minoritaria = classes_unicas[np.argmin(contagens)]
        classe_majoritaria = classes_unicas[np.argmax(contagens)]

        n_minoritaria = np.sum(rótulos == classe_minoritaria)
        n_majoritaria = np.sum(rótulos == classe_majoritaria)

        # Calcular número de amostras sintéticas a gerar
        n_sintetizar = n_majoritaria - n_minoritaria

        if n_sintetizar <= 0:
            self.logger.warning("Dataset já balanceado ou minoritária maior")
            self.registrar_distribuicao(rótulos, rótulos, "SMOTE")
            return imagens, rótulos

        # Selecionar amostras minoritárias
        indices_minoritaria = np.where(rótulos == classe_minoritaria)[0]
        imagens_minoritaria = imagens_2d[indices_minoritaria]

        # Usar k-NN para encontrar vizinhos
        knn = NearestNeighbors(n_neighbors=self.n_vizinhos + 1, n_jobs=self.n_trabalhos)
        knn.fit(imagens_minoritaria)
        _, indices_vizinhos = knn.kneighbors(imagens_minoritaria)

        # Gerar amostras sintéticas
        imagens_sinteticas = []
        rótulos_sinteticos = []

        for i in range(n_sintetizar):
            # Selecionar amostra minoritária aleatória
            idx_amostra = np.random.randint(0, len(imagens_minoritaria))
            amostra = imagens_minoritaria[idx_amostra]

            # Selecionar vizinho aleatório (excluir o próprio - índice 0)
            idx_vizinho = np.random.randint(1, self.n_vizinhos + 1)
            vizinho = imagens_minoritaria[indices_vizinhos[idx_amostra, idx_vizinho]]

            # Interpolar entre amostra e vizinho
            alpha = np.random.rand()
            amostra_sintetica = amostra + alpha * (vizinho - amostra)

            imagens_sinteticas.append(amostra_sintetica)
            rótulos_sinteticos.append(classe_minoritaria)

        # Combinar com dataset original
        imagens_sinteticas = np.array(imagens_sinteticas)
        rótulos_sinteticos = np.array(rótulos_sinteticos)

        imagens_balanceadas_2d = np.vstack([imagens_2d, imagens_sinteticas])
        rótulos_balanceados = np.hstack([rótulos, rótulos_sinteticos])

        # Remodelar de volta para 4D
        imagens_balanceadas = imagens_balanceadas_2d.reshape(
            -1, altura, largura, canais
        )

        self.registrar_distribuicao(rótulos, rótulos_balanceados, "SMOTE")

        return imagens_balanceadas, rótulos_balanceados
