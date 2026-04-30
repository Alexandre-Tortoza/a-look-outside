"""Arquitetura CNN Baseline com blocos residuais."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from modelos.base import ClassificadorGalaxias
from utils.xai_gradcam import grad_cam


class _BlocoResidual(nn.Module):
    """Bloco conv 3×3 → BN → ReLU → conv 3×3 → BN com conexão skip."""

    def __init__(self, canais: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(canais, canais, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(canais)
        self.conv2 = nn.Conv2d(canais, canais, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(canais)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residuo = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residuo)


class RedeCNNBaseline(nn.Module):
    """CNN de 5 blocos com conexões residuais e GlobalAveragePooling.

    Arquitetura:
        Conv(3→64) BN ReLU MaxPool(2)
        Conv(64→128) BN ReLU MaxPool(2)
        Conv(128→256) BN ReLU
        ResBlock(256) MaxPool(2)
        ResBlock(256→512, projeção) MaxPool(2)
        AdaptiveAvgPool2d(1) → Linear(512, num_classes)

    O AdaptiveAvgPool garante compatibilidade com qualquer tamanho de entrada.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.bloco1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.bloco2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.bloco3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.bloco4 = nn.Sequential(
            _BlocoResidual(256),
            nn.MaxPool2d(2),
        )
        # Projeção 256→512 para o bloco residual final
        self.projecao = nn.Sequential(
            nn.Conv2d(256, 512, 1, bias=False),
            nn.BatchNorm2d(512),
        )
        self.bloco5 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
        )
        self.relu = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.classificador = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bloco1(x)
        x = self.bloco2(x)
        x = self.bloco3(x)
        x = self.bloco4(x)
        residuo = self.projecao(x)
        x = self.relu(self.bloco5(x) + residuo)
        x = self.pool5(x)
        x = self.gap(x)
        x = self.dropout(x.flatten(1))
        return self.classificador(x)


class CNNBaseline(ClassificadorGalaxias):
    """Wrapper ClassificadorGalaxias para a RedeCNNBaseline."""

    @property
    def nome(self) -> str:
        return "CNN"

    @property
    def variante(self) -> str:
        return "baseline"

    @property
    def metodo_xai(self) -> str:
        return "grad-cam"

    @property
    def camadas_xai(self) -> list[str]:
        # Última camada conv antes do GAP (segundo conv do bloco5)
        return ["bloco5.3"]

    def construir(self, num_classes: int = 10, tamanho_imagem: int = 224) -> RedeCNNBaseline:
        return RedeCNNBaseline(num_classes=num_classes)

    def explicar(
        self,
        rede: nn.Module,
        tensor_entrada: torch.Tensor,
        classe_alvo: Optional[int] = None,
    ) -> np.ndarray:
        camada = self.camadas_xai[0]
        h, w = tensor_entrada.shape[2], tensor_entrada.shape[3]
        return grad_cam(rede, tensor_entrada, camada, classe_alvo, tamanho_saida=(h, w))
