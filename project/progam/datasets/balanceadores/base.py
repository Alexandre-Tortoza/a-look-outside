"""
Classe base abstrata para balanceadores de datasets.

Define interface comum para todas as estratégias de balanceamento.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BalanceadorBase(ABC):
    """Classe base para balanceadores de datasets."""

    def __init__(self, semente: Optional[int] = None):
        """
        Inicializar balanceador.

        Args:
            semente: Seed para reprodutibilidade.
        """
        self.semente = semente
        self.logger = logger

        if semente is not None:
            np.random.seed(semente)

    @abstractmethod
    def balancear(
        self,
        imagens: np.ndarray,
        rótulos: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aplicar técnica de balanceamento.

        Args:
            imagens: Array com imagens [n_amostras, altura, largura, canais].
            rótulos: Array com rótulos [n_amostras].

        Returns:
            Tupla (imagens_balanceadas, rótulos_balanceados).
        """
        pass

    def obter_distribuicao_classes(self, rótulos: np.ndarray) -> dict:
        """
        Obter distribuição de classes nos rótulos.

        Args:
            rótulos: Array com rótulos.

        Returns:
            Dicionário com contagem por classe.
        """
        classes_unicas, contagens = np.unique(rótulos, return_counts=True)
        return {classe: int(contagem) for classe, contagem in zip(classes_unicas, contagens)}

    def registrar_distribuicao(
        self,
        rótulos_antes: np.ndarray,
        rótulos_depois: np.ndarray,
        nome_metodo: str,
    ) -> None:
        """
        Registrar distribuição antes e depois do balanceamento.

        Args:
            rótulos_antes: Rótulos originais.
            rótulos_depois: Rótulos após balanceamento.
            nome_metodo: Nome do método de balanceamento.
        """
        dist_antes = self.obter_distribuicao_classes(rótulos_antes)
        dist_depois = self.obter_distribuicao_classes(rótulos_depois)

        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"Balanceamento: {nome_metodo}")
        self.logger.info(f"{'=' * 60}")
        self.logger.info(f"Distribuição ANTES: {dist_antes}")
        self.logger.info(f"Distribuição DEPOIS: {dist_depois}")
        self.logger.info(f"Total de amostras: {len(rótulos_antes)} → {len(rótulos_depois)}")
