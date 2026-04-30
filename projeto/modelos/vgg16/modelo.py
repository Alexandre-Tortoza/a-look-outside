"""VGG16 com fine-tuning via timm."""

from __future__ import annotations

from typing import Optional

import numpy as np
import timm
import torch
import torch.nn as nn

from modelos.base import ClassificadorGalaxias
from utils.xai_gradcam import grad_cam


class VGG16Galaxy(ClassificadorGalaxias):
    """VGG16 pré-treinado no ImageNet, ajustado para Galaxy10."""

    def __init__(self, pretrained: bool = True) -> None:
        self.pretrained = pretrained

    @property
    def nome(self) -> str:
        return "VGG16"

    @property
    def variante(self) -> str:
        return "pretrained" if self.pretrained else "scratch"

    @property
    def metodo_xai(self) -> str:
        return "grad-cam"

    @property
    def camadas_xai(self) -> list[str]:
        # Último bloco conv (5.º bloco, 3.ª conv) do VGG16
        return ["features.28"]

    @property
    def suporta_finetune(self) -> bool:
        return True

    def construir(self, num_classes: int = 10, tamanho_imagem: int = 224) -> nn.Module:
        return timm.create_model(
            "vgg16",
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
