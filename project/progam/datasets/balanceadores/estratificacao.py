"""
Balanceador Estratificação.

Mantém proporções estratificadas das classes ao subamostrar ou sobreamostrar.
"""

from typing import Optional, Tuple

import numpy as np

from .base import BalanceadorBase


class BalanceadorEstratificacao(BalanceadorBase):
    """Implementação de Estratificação para balanceamento de datasets."""

    def __init__(
        self,
        fracao: float = 0.8,
        semente: Optional[int] = None,
    ):
        """
        Inicializar balanceador Estratificação.

        Args:
            fracao: Fração do dataset a manter (0.0 a 1.0).
            semente: Seed para reprodutibilidade.
        """
        super().__init__(semente=semente)
        self.fracao = max(0.0, min(1.0, fracao))

    def balancear(
        self,
        imagens: np.ndarray,
        rótulos: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplicar Estratificação.

        Mantém proporções de classes ao realizar subamostragem.

        Args:
            imagens: Array com imagens [n_amostras, altura, largura, canais].
            rótulos: Array com rótulos [n_amostras].

        Returns:
            Tupla (imagens_balanceadas, rótulos_balanceados).
        """
        self.logger.info(f"Aplicando Estratificação (fração: {self.fracao})...")

        # Obter distribuição original
        classes_unicas = np.unique(rótulos)

        # Calcular número de amostras a manter por classe
        n_total_manter = int(len(rótulos) * self.fracao)
        indices_manter = []

        for classe in classes_unicas:
            indices_classe = np.where(rótulos == classe)[0]
            n_classe = len(indices_classe)

            # Calcular proporção dessa classe no dataset original
            proporcao_classe = n_classe / len(rótulos)

            # Calcular quantas amostras dessa classe devem ser mantidas
            n_classe_manter = int(n_total_manter * proporcao_classe)

            # Subamostrar mantendo proporção
            indices_subamostrados = np.random.choice(
                indices_classe,
                size=n_classe_manter,
                replace=False,
            )
            indices_manter.extend(indices_subamostrados)

        # Ordenar índices
        indices_manter = np.sort(indices_manter)

        # Aplicar seleção
        imagens_balanceadas = imagens[indices_manter]
        rótulos_balanceados = rótulos[indices_manter]

        self.registrar_distribuicao(rótulos, rótulos_balanceados, "Estratificação")

        return imagens_balanceadas, rótulos_balanceados
