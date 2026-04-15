"""Arquiteturas MobileNet para classificação de morfologia galáctica."""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from models.base import GalaxyClassifier


class _ConvSeparavelProfundidade(nn.Module):
    """Convolução depthwise-separável: depthwise + pointwise.

    Reduz significativamente o número de parâmetros mantendo
    a capacidade expressiva.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        # Depthwise: aplica convolução separadamente para cada canal
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch)
        self.bn_dw = nn.BatchNorm2d(in_ch)
        # Pointwise: combina os canais
        self.pw = nn.Conv2d(in_ch, out_ch, 1)
        self.bn_pw = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn_dw(self.dw(x)))
        return self.relu(self.bn_pw(self.pw(x)))


class _BlocoResidualInvertido(nn.Module):
    """Bloco residual invertido (MobileNetV2-style).

    Expande dimensionalidade → depthwise → reduz dimensionalidade.
    Skip connection apenas quando stride=1 e in_ch==out_ch.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int, fator_expansao: int):
        super().__init__()
        oculto = in_ch * fator_expansao
        self.usa_skip = (stride == 1 and in_ch == out_ch)

        camadas = []
        # Expansão (se fator > 1)
        if fator_expansao != 1:
            camadas.extend([
                nn.Conv2d(in_ch, oculto, 1),
                nn.BatchNorm2d(oculto),
                nn.ReLU6(inplace=True),
            ])
        # Depthwise
        camadas.extend([
            nn.Conv2d(oculto, oculto, 3, stride=stride, padding=1, groups=oculto),
            nn.BatchNorm2d(oculto),
            nn.ReLU6(inplace=True),
        ])
        # Compressão
        camadas.extend([
            nn.Conv2d(oculto, out_ch, 1),
            nn.BatchNorm2d(out_ch),
        ])
        self.conv = nn.Sequential(*camadas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x) if self.usa_skip else self.conv(x)


class RedeMovelSimples(nn.Module):
    """MobileNet leve com multiplicador de largura 0.5.

    Arquitetura:
    - Convolução inicial (3→16, stride=2)
    - 4 blocos depthwise-separáveis (stride=2 cada)
    - AdaptiveAvgPool2d → Dense(num_classes)

    Otimizada para edge/mobile com baixa latência.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.inicial = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.blocos = nn.Sequential(
            _ConvSeparavelProfundidade(16, 32, stride=2),
            _ConvSeparavelProfundidade(32, 64, stride=2),
            _ConvSeparavelProfundidade(64, 64, stride=2),
            _ConvSeparavelProfundidade(64, 128, stride=2),
        )
        self.pool_global = nn.AdaptiveAvgPool2d(1)
        self.classificador = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inicial(x)
        x = self.blocos(x)
        x = self.pool_global(x).flatten(1)
        return self.classificador(x)


class RedeMovelRobusta(nn.Module):
    """MobileNet robusta com multiplicador de largura 1.0 e blocos residuais invertidos.

    Arquitetura:
    - Convolução inicial (3→32, stride=2)
    - 6 blocos residuais invertidos (MobileNetV2-style)
    - AdaptiveAvgPool2d → Dense(1024) → ReLU → Dropout(0.2) → Dense(num_classes)

    Equilibra eficiência e precisão para produção.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.inicial = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )
        self.blocos = nn.Sequential(
            _BlocoResidualInvertido(32, 16, stride=1, fator_expansao=1),
            _BlocoResidualInvertido(16, 24, stride=2, fator_expansao=6),
            _BlocoResidualInvertido(24, 32, stride=2, fator_expansao=6),
            _BlocoResidualInvertido(32, 64, stride=2, fator_expansao=6),
            _BlocoResidualInvertido(64, 96, stride=1, fator_expansao=6),
            _BlocoResidualInvertido(96, 160, stride=2, fator_expansao=6),
        )
        self.pool_global = nn.AdaptiveAvgPool2d(1)
        self.classificador = nn.Sequential(
            nn.Linear(160, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inicial(x)
        x = self.blocos(x)
        x = self.pool_global(x).flatten(1)
        return self.classificador(x)


class MobileNetLight(GalaxyClassifier):
    """Wrapper MobileNet Simples para orquestração de benchmark."""

    @property
    def name(self) -> str:
        return "MobileNet"

    @property
    def variant(self) -> str:
        return "light"

    @property
    def xai_method(self) -> str:
        return "grad-cam"

    def build(self, num_classes: int, img_size: int) -> nn.Module:
        """Instancia a rede MobileNet simples."""
        return RedeMovelSimples(num_classes)

    def explain(self, model, input_tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Retorna mapa Grad-CAM (stub)."""
        h, w = input_tensor.shape[-2], input_tensor.shape[-1]
        return np.zeros((h, w), dtype=np.float32)


class MobileNetRobust(GalaxyClassifier):
    """Wrapper MobileNet Robusta para orquestração de benchmark."""

    @property
    def name(self) -> str:
        return "MobileNet"

    @property
    def variant(self) -> str:
        return "robust"

    @property
    def xai_method(self) -> str:
        return "grad-cam"

    def build(self, num_classes: int, img_size: int) -> nn.Module:
        """Instancia a rede MobileNet robusta."""
        return RedeMovelRobusta(num_classes)

    def explain(self, model, input_tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Retorna mapa Grad-CAM (stub)."""
        h, w = input_tensor.shape[-2], input_tensor.shape[-1]
        return np.zeros((h, w), dtype=np.float32)
