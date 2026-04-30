"""Vision Transformer (ViT-B/16) com Attention Rollout para XAI."""

from __future__ import annotations

from typing import Optional

import numpy as np
import timm
import torch
import torch.nn as nn

from modelos.base import ClassificadorGalaxias
from modelos.vit.xai import attention_rollout


class ViTGalaxy(ClassificadorGalaxias):
    """ViT-Base/16 pré-treinado no ImageNet-21k, ajustado para Galaxy10."""

    def __init__(
        self,
        backbone: str = "vit_base_patch16_224",
        pretrained: bool = True,
    ) -> None:
        self.backbone = backbone
        self.pretrained = pretrained

    @property
    def nome(self) -> str:
        return "ViT"

    @property
    def variante(self) -> str:
        return f"{self.backbone}_{'pretrained' if self.pretrained else 'scratch'}"

    @property
    def metodo_xai(self) -> str:
        return "attention-rollout"

    @property
    def camadas_xai(self) -> list[str]:
        return ["blocks"]  # lista dos transformer blocks

    @property
    def suporta_finetune(self) -> bool:
        return True

    def construir(self, num_classes: int = 10, tamanho_imagem: int = 224) -> nn.Module:
        return timm.create_model(
            self.backbone,
            pretrained=self.pretrained,
            num_classes=num_classes,
            img_size=tamanho_imagem,
        )

    def explicar(
        self,
        rede: nn.Module,
        tensor_entrada: torch.Tensor,
        classe_alvo: Optional[int] = None,
    ) -> np.ndarray:
        h, w = tensor_entrada.shape[2], tensor_entrada.shape[3]
        return attention_rollout(rede, tensor_entrada, tamanho_saida=(h, w))
