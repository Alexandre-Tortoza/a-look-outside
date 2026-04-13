"""Arquiteturas EasyNet para classificação de morfologia galáctica."""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from models.base import GalaxyClassifier


class RedeEasyNetSimples(nn.Module):
    """Rede EasyNet leve com 2 blocos convolucionais.

    Arquitetura:
    - Conv(3→32) → ReLU → MaxPool(2)
    - Conv(32→64) → ReLU → MaxPool(2)
    - AdaptiveAvgPool2d → Dense(64, num_classes)

    Modelo base, otimizado para prototipagem rápida.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.extrator_caracteristicas = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool_global = nn.AdaptiveAvgPool2d(1)
        self.classificador = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extrator_caracteristicas(x)
        x = self.pool_global(x).flatten(1)
        return self.classificador(x)


class RedeEasyNetRobusta(nn.Module):
    """Rede EasyNet robusta com 3 blocos convolucionais.

    Arquitetura:
    - Conv(3→32) → BN → ReLU → MaxPool(2)
    - Conv(32→64) → BN → ReLU → MaxPool(2)
    - Conv(64→128) → BN → ReLU → MaxPool(2)
    - AdaptiveAvgPool2d → Dense(128, 128) → ReLU → Dropout → Dense(128, num_classes)

    Versão melhorada com BatchNorm e dropout para melhor generalização.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.extrator_caracteristicas = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool_global = nn.AdaptiveAvgPool2d(1)
        self.classificador = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extrator_caracteristicas(x)
        x = self.pool_global(x).flatten(1)
        return self.classificador(x)


class EasyNet(GalaxyClassifier):
    """Wrapper EasyNet para orquestração de benchmark.

    Modelo leve, adequado para experimentos rápidos e linha de base.
    """

    @property
    def name(self) -> str:
        return "EasyNet"

    @property
    def variant(self) -> str:
        return "light"

    @property
    def xai_method(self) -> str:
        return "lime"

    def build(self, num_classes: int, img_size: int) -> nn.Module:
        """Instancia a rede EasyNet simples."""
        return RedeEasyNetSimples(num_classes)

    def explain(self, model, input_tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Retorna mapa de explicação LIME (stub)."""
        h, w = input_tensor.shape[-2], input_tensor.shape[-1]
        return np.zeros((h, w), dtype=np.float32)


class EasyNetRobust(GalaxyClassifier):
    """Wrapper EasyNet Robusta para orquestração de benchmark.

    Modelo melhorado com BatchNorm e dropout para melhor
    generalização cross-dataset.
    """

    @property
    def name(self) -> str:
        return "EasyNet"

    @property
    def variant(self) -> str:
        return "robust"

    @property
    def xai_method(self) -> str:
        return "lime"

    def build(self, num_classes: int, img_size: int) -> nn.Module:
        """Instancia a rede EasyNet robusta."""
        return RedeEasyNetRobusta(num_classes)

    def explain(self, model, input_tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Retorna mapa de explicação LIME (stub)."""
        h, w = input_tensor.shape[-2], input_tensor.shape[-1]
        return np.zeros((h, w), dtype=np.float32)
