"""Contrato abstrato para todos os classificadores de galáxias."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class ClassificadorGalaxias(ABC):
    """ABC que todos os sete modelos do projeto devem implementar.

    Cada subclasse encapsula:
    - A construção da arquitetura PyTorch (construir).
    - A geração de explicações XAI para uma amostra (explicar).
    - Metadados do modelo (nome, variante, metodo_xai, camadas_xai).
    """

    # ------------------------------------------------------------------
    # Metadados obrigatórios
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def nome(self) -> str:
        """Identificador curto, ex: 'CNN', 'ResNet50', 'ViT'."""

    @property
    @abstractmethod
    def variante(self) -> str:
        """Descrição da variante, ex: 'baseline', 'pretrained', 'self-supervised'."""

    @property
    @abstractmethod
    def metodo_xai(self) -> str:
        """Técnica XAI usada: 'grad-cam', 'attention-rollout', 'attention-maps', 'shap'."""

    @property
    def camadas_xai(self) -> list[str]:
        """Nomes de camadas relevantes para XAI em notação de pontos.

        Subclasses que usam Grad-CAM devem sobrescrever com a última camada conv.
        """
        return []

    # ------------------------------------------------------------------
    # Capacidades (override conforme necessário)
    # ------------------------------------------------------------------

    @property
    def suporta_checkpoint(self) -> bool:
        """Se o modelo suporta salvar/carregar checkpoints."""
        return True

    @property
    def suporta_finetune(self) -> bool:
        """Se o modelo suporta fine-tuning a partir de pesos pré-treinados."""
        return False

    # ------------------------------------------------------------------
    # Métodos principais
    # ------------------------------------------------------------------

    @abstractmethod
    def construir(self, num_classes: int, tamanho_imagem: int) -> nn.Module:
        """Instancia e retorna o módulo PyTorch pronto para treino.

        Args:
            num_classes: Número de classes de saída.
            tamanho_imagem: Tamanho da imagem de entrada (quadrada).

        Returns:
            nn.Module configurado.
        """

    @abstractmethod
    def explicar(
        self,
        rede: nn.Module,
        tensor_entrada: torch.Tensor,
        classe_alvo: Optional[int] = None,
    ) -> np.ndarray:
        """Gera mapa de saliência XAI para uma amostra.

        Args:
            rede: Modelo treinado em modo eval.
            tensor_entrada: Tensor (1, C, H, W) normalizado.
            classe_alvo: Classe a explicar. Se None, usa a predita.

        Returns:
            Array float32 (H, W) normalizado em [0, 1].
        """

    def __repr__(self) -> str:
        return f"{self.nome}[{self.variante}]({self.metodo_xai})"
