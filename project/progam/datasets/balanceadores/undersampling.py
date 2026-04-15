"""
Balanceador Undersampling.

Remove amostras da classe majoritária até atingir equilíbrio.
"""

from typing import Optional, Tuple

import numpy as np

from .base import BalanceadorBase


class BalanceadorUndersampling(BalanceadorBase):
    """Implementação de Undersampling para balanceamento de datasets."""

    def __init__(self, semente: Optional[int] = None):
        """
        Inicializar balanceador Undersampling.

        Args:
            semente: Seed para reprodutibilidade.
        """
        super().__init__(semente=semente)

    def balancear(
        self,
        imagens: np.ndarray,
        rótulos: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplicar Undersampling.

        Args:
            imagens: Array com imagens [n_amostras, altura, largura, canais].
            rótulos: Array com rótulos [n_amostras].

        Returns:
            Tupla (imagens_balanceadas, rótulos_balanceados).
        """
        self.logger.info("Aplicando Undersampling...")

        # Obter distribuição original
        classes_unicas, contagens = np.unique(rótulos, return_counts=True)
        classe_minoritaria = classes_unicas[np.argmin(contagens)]

        n_minoritaria = np.sum(rótulos == classe_minoritaria)

        # Selecionar amostras para manter
        indices_manter = []

        # Manter todas as amostras minoritárias
        indices_manter.extend(np.where(rótulos == classe_minoritaria)[0])

        # Subamostrar classe majoritária
        for classe in classes_unicas:
            if classe != classe_minoritaria:
                indices_classe = np.where(rótulos == classe)[0]
                indices_subamostrados = np.random.choice(
                    indices_classe,
                    size=n_minoritaria,
                    replace=False,
                )
                indices_manter.extend(indices_subamostrados)

        # Ordenar índices para manter ordem
        indices_manter = np.sort(indices_manter)

        # Aplicar seleção
        imagens_balanceadas = imagens[indices_manter]
        rótulos_balanceados = rótulos[indices_manter]

        self.registrar_distribuicao(rótulos, rótulos_balanceados, "Undersampling")

        return imagens_balanceadas, rótulos_balanceados
