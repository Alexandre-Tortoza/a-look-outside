"""
Balanceador Oversampling (Random).

Duplica aleatoriamente amostras da classe minoritária até atingir equilíbrio.
"""

from typing import Optional, Tuple

import numpy as np

from .base import BalanceadorBase


class BalanceadorOversampling(BalanceadorBase):
    """Implementação de Oversampling aleatório para balanceamento de datasets."""

    def __init__(self, semente: Optional[int] = None):
        """
        Inicializar balanceador Oversampling.

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
        Aplicar Oversampling.

        Args:
            imagens: Array com imagens [n_amostras, altura, largura, canais].
            rótulos: Array com rótulos [n_amostras].

        Returns:
            Tupla (imagens_balanceadas, rótulos_balanceados).
        """
        self.logger.info("Aplicando Oversampling...")

        # Obter distribuição original
        classes_unicas, contagens = np.unique(rótulos, return_counts=True)
        classe_majoritaria = classes_unicas[np.argmax(contagens)]
        n_majoritaria = np.max(contagens)

        # Selecionar amostras para manter/duplicar
        indices_manter = []

        # Manter todas as amostras da classe majoritária
        indices_manter.extend(np.where(rótulos == classe_majoritaria)[0])

        # Sobreamostrar classes minoritárias
        for classe in classes_unicas:
            if classe != classe_majoritaria:
                indices_classe = np.where(rótulos == classe)[0]
                indices_duplicados = np.random.choice(
                    indices_classe,
                    size=n_majoritaria,
                    replace=True,
                )
                indices_manter.extend(indices_duplicados)

        # Misturar índices
        indices_manter = np.array(indices_manter)
        np.random.shuffle(indices_manter)

        # Aplicar seleção
        imagens_balanceadas = imagens[indices_manter]
        rótulos_balanceados = rótulos[indices_manter]

        self.registrar_distribuicao(rótulos, rótulos_balanceados, "Oversampling")

        return imagens_balanceadas, rótulos_balanceados
