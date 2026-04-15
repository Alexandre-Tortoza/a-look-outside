"""Arquiteturas CNN clássicas para classificação de morfologia galáctica."""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from models.base import GalaxyClassifier


class _BlocoResidual(nn.Module):
    """Bloco residual com duas convoluções e skip connection.

    Se in_ch != out_ch, aplica convolução 1x1 no skip para ajustar dimensões.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residuo = self.skip(x)
        saida = self.relu(self.bn1(self.conv1(x)))
        saida = self.bn2(self.conv2(saida))
        return self.relu(saida + residuo)


class RedeCNNSimples(nn.Module):
    """Rede CNN leve com 3 blocos convolucionais.

    Arquitetura:
    - Conv(3→32) → ReLU → MaxPool(2)
    - Conv(32→64) → ReLU → MaxPool(2)
    - Conv(64→128) → ReLU → MaxPool(2)
    - Flatten → Dense(256) → ReLU → Dropout(0.3) → Dense(num_classes)

    Otimizada para velocidade de inferência.
    """

    def __init__(self, tamanho_flat: int, num_classes: int):
        super().__init__()
        self.extrator_caracteristicas = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classificador = nn.Sequential(
            nn.Linear(tamanho_flat, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extrator_caracteristicas(x)
        x = x.flatten(1)
        return self.classificador(x)


class RedeCNNRobusta(nn.Module):
    """Rede CNN robusta com 5 blocos e conexões residuais.

    Arquitetura:
    - Conv(3→64) → BN → ReLU → MaxPool(2)
    - Conv(64→128) → BN → ReLU → MaxPool(2)
    - Conv(128→256) → BN → ReLU
    - BlocoResidual(256→256) → MaxPool(2)
    - BlocoResidual(256→512) → MaxPool(2)
    - Flatten → Dense(512) → ReLU → Dropout(0.5)
               → Dense(256) → ReLU → Dropout(0.4) → Dense(num_classes)

    Otimizada para máxima precisão cross-dataset.
    """

    def __init__(self, tamanho_flat: int, num_classes: int):
        super().__init__()
        self.bloco1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.bloco2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.bloco3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.bloco4 = nn.Sequential(
            _BlocoResidual(256, 256),
            nn.MaxPool2d(2),
        )
        self.bloco5 = nn.Sequential(
            _BlocoResidual(256, 512),
            nn.MaxPool2d(2),
        )
        self.classificador = nn.Sequential(
            nn.Linear(tamanho_flat, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bloco1(x)
        x = self.bloco2(x)
        x = self.bloco3(x)
        x = self.bloco4(x)
        x = self.bloco5(x)
        x = x.flatten(1)
        return self.classificador(x)


class CNNLight(GalaxyClassifier):
    """Wrapper CNN Simples para orquestração de benchmark."""

    @property
    def name(self) -> str:
        return "CNN"

    @property
    def variant(self) -> str:
        return "light"

    @property
    def xai_method(self) -> str:
        return "grad-cam"

    def build(self, num_classes: int, img_size: int) -> nn.Module:
        """Instancia a rede CNN simples."""
        tamanho_flat = 128 * (img_size // 8) * (img_size // 8)
        return RedeCNNSimples(tamanho_flat, num_classes)

    def explain(self, model, input_tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Retorna mapa Grad-CAM (stub)."""
        h, w = input_tensor.shape[-2], input_tensor.shape[-1]
        return np.zeros((h, w), dtype=np.float32)


class CNNRobust(GalaxyClassifier):
    """Wrapper CNN Robusta para orquestração de benchmark."""

    @property
    def name(self) -> str:
        return "CNN"

    @property
    def variant(self) -> str:
        return "robust"

    @property
    def xai_method(self) -> str:
        return "grad-cam"

    def build(self, num_classes: int, img_size: int) -> nn.Module:
        """Instancia a rede CNN robusta."""
        tamanho_flat = 512 * (img_size // 16) * (img_size // 16)
        return RedeCNNRobusta(tamanho_flat, num_classes)

    def explain(self, model, input_tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Retorna mapa Grad-CAM (stub)."""
        h, w = input_tensor.shape[-2], input_tensor.shape[-1]
        return np.zeros((h, w), dtype=np.float32)
