"""
Balanceador ADASYN (Adaptive Synthetic Sampling).

Gera amostras sintéticas adaptativas focando em regiões de decisão difíceis.
"""

from typing import Optional, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .base import BalanceadorBase


class BalanceadorADASYN(BalanceadorBase):
    """Implementação de ADASYN para balanceamento de datasets."""

    def __init__(
        self,
        n_vizinhos: int = 5,
        n_trabalhos: int = -1,
        semente: Optional[int] = None,
    ):
        """
        Inicializar balanceador ADASYN.

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
        Aplicar ADASYN.

        Args:
            imagens: Array com imagens [n_amostras, altura, largura, canais].
            rótulos: Array com rótulos [n_amostras].

        Returns:
            Tupla (imagens_balanceadas, rótulos_balanceados).
        """
        self.logger.info("Aplicando ADASYN...")

        # Remodelar imagens para 2D (n_amostras, features)
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
            self.registrar_distribuicao(rótulos, rótulos, "ADASYN")
            return imagens, rótulos

        # Selecionar amostras minoritárias
        indices_minoritaria = np.where(rótulos == classe_minoritaria)[0]
        imagens_minoritaria = imagens_2d[indices_minoritaria]

        # Usar k-NN para encontrar vizinhos
        knn = NearestNeighbors(n_neighbors=self.n_vizinhos + 1, n_jobs=self.n_trabalhos)
        knn.fit(imagens_2d)

        # Calcular densidade de vizinhos da classe majoritária
        densidades = np.zeros(len(indices_minoritaria))

        for i, idx in enumerate(indices_minoritaria):
            _, indices_vizinhos = knn.kneighbors([imagens_2d[idx]])
            vizinhos = rótulos[indices_vizinhos[0]]
            n_majoritaria_vizinhos = np.sum(vizinhos == classe_majoritaria)
            densidades[i] = n_majoritaria_vizinhos / self.n_vizinhos

        # Normalizar densidades para pesos
        densidades = densidades / np.sum(densidades)
        n_sintetizar_por_amostra = (densidades * n_sintetizar).astype(int)

        # Gerar amostras sintéticas
        imagens_sinteticas = []
        rótulos_sinteticos = []

        for i, idx in enumerate(indices_minoritaria):
            vizinho_indices = np.random.choice(
                len(imagens_minoritaria),
                size=n_sintetizar_por_amostra[i],
                replace=True,
            )

            for vizinho_idx in vizinho_indices:
                # Interpolar entre amostra e vizinho
                alpha = np.random.rand()
                amostra_sintetica = (
                    imagens_minoritaria[i] * alpha
                    + imagens_minoritaria[vizinho_idx] * (1 - alpha)
                )
                imagens_sinteticas.append(amostra_sintetica)
                rótulos_sinteticos.append(classe_minoritaria)

        # Combinar com dataset original
        if imagens_sinteticas:
            imagens_sinteticas = np.array(imagens_sinteticas)
            rótulos_sinteticos = np.array(rótulos_sinteticos)

            imagens_balanceadas_2d = np.vstack([imagens_2d, imagens_sinteticas])
            rótulos_balanceados = np.hstack([rótulos, rótulos_sinteticos])
        else:
            imagens_balanceadas_2d = imagens_2d
            rótulos_balanceados = rótulos

        # Remodelar de volta para 4D
        imagens_balanceadas = imagens_balanceadas_2d.reshape(
            -1, altura, largura, canais
        )

        self.registrar_distribuicao(rótulos, rótulos_balanceados, "ADASYN")

        return imagens_balanceadas, rótulos_balanceados
