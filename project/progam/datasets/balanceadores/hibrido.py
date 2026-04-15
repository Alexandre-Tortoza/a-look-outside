"""
Balanceador Híbrido.

Combina múltiplas técnicas de balanceamento (ex: SMOTE + Undersampling).
"""

from typing import List, Optional, Tuple

import numpy as np

from .base import BalanceadorBase


class BalanceadorHibrido(BalanceadorBase):
    """Implementação de Híbrido para balanceamento de datasets."""

    def __init__(
        self,
        balanceadores: Optional[List[BalanceadorBase]] = None,
        semente: Optional[int] = None,
    ):
        """
        Inicializar balanceador Híbrido.

        Args:
            balanceadores: Lista de balanceadores a aplicar em sequência.
                         Se None, aplica SMOTE + Undersampling padrão.
            semente: Seed para reprodutibilidade.
        """
        super().__init__(semente=semente)

        if balanceadores is None:
            # Combinação padrão: SMOTE + Undersampling
            from .smote import BalanceadorSMOTE
            from .undersampling import BalanceadorUndersampling

            balanceadores = [
                BalanceadorSMOTE(semente=semente),
                BalanceadorUndersampling(semente=semente),
            ]

        self.balanceadores = balanceadores

    def balancear(
        self,
        imagens: np.ndarray,
        rótulos: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplicar Híbrido (múltiplas técnicas em sequência).

        Args:
            imagens: Array com imagens [n_amostras, altura, largura, canais].
            rótulos: Array com rótulos [n_amostras].

        Returns:
            Tupla (imagens_balanceadas, rótulos_balanceados).
        """
        self.logger.info("Aplicando Balanceamento Híbrido...")

        imagens_atual = imagens
        rótulos_atual = rótulos

        # Aplicar cada balanceador em sequência
        for i, balanceador in enumerate(self.balanceadores, 1):
            self.logger.info(f"Etapa {i}/{len(self.balanceadores)}: {balanceador.__class__.__name__}")
            imagens_atual, rótulos_atual = balanceador.balancear(
                imagens_atual,
                rótulos_atual,
            )

        self.registrar_distribuicao(rótulos, rótulos_atual, "Híbrido")

        return imagens_atual, rótulos_atual
