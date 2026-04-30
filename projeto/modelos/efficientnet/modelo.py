"""EfficientNet-B0 com fine-tuning via timm."""

from __future__ import annotations

from typing import Optional

import numpy as np
import timm
import torch
import torch.nn as nn

from modelos.base import ClassificadorGalaxias
from utils.xai_gradcam import grad_cam


class EfficientNetGalaxy(ClassificadorGalaxias):
    """EfficientNet-B0 (ou outra variante) pré-treinado no ImageNet."""

    def __init__(
        self,
        variante_backbone: str = "efficientnet_b0",
        pretrained: bool = True,
    ) -> None:
        self.variante_backbone = variante_backbone
        self.pretrained = pretrained

    @property
    def nome(self) -> str:
        return "EfficientNet"

    @property
    def variante(self) -> str:
        return f"{self.variante_backbone}_{'pretrained' if self.pretrained else 'scratch'}"

    @property
    def metodo_xai(self) -> str:
        return "grad-cam"

    @property
    def camadas_xai(self) -> list[str]:
        # Último bloco conv do EfficientNet-B0
        return ["blocks.6.0.conv_pwl"]

    @property
    def suporta_finetune(self) -> bool:
        return True

    def construir(self, num_classes: int = 10, tamanho_imagem: int = 224) -> nn.Module:
        return timm.create_model(
            self.variante_backbone,
            pretrained=self.pretrained,
            num_classes=num_classes,
        )

    def explicar(
        self,
        rede: nn.Module,
        tensor_entrada: torch.Tensor,
        classe_alvo: Optional[int] = None,
    ) -> np.ndarray:
        h, w = tensor_entrada.shape[2], tensor_entrada.shape[3]
        return grad_cam(rede, tensor_entrada, self.camadas_xai[0], classe_alvo, (h, w))
