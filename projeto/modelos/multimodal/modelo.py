"""Modelo Multimodal: fusão de branch visual (EfficientNet) + branch tabular (MLP)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import timm
import torch
import torch.nn as nn

from modelos.base import ClassificadorGalaxias
from modelos.multimodal import config as cfg


class _BranchVisual(nn.Module):
    """EfficientNet-B0 como extrator visual. Retorna vetor de features."""

    def __init__(self, pretrained: bool = True, dim_saida: int = 256) -> None:
        super().__init__()
        backbone = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0)
        dim_backbone = backbone.num_features
        self.backbone = backbone
        self.projecao = nn.Sequential(
            nn.Linear(dim_backbone, dim_saida),
            nn.BatchNorm1d(dim_saida),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.projecao(features)


class _BranchTabular(nn.Module):
    """MLP para features tabulares."""

    def __init__(
        self,
        num_features: int,
        dim_saida: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, dim_saida),
            nn.BatchNorm1d(dim_saida),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class RedeMultimodal(nn.Module):
    """Fusão tardia: branch visual + branch tabular → classificador.

    Forward: (imagem, tabular) → logits
    """

    def __init__(
        self,
        num_classes: int = 10,
        num_features_tabulares: int = 6,
        pretrained: bool = True,
        dim_visual: int = cfg.DIM_VISUAL,
        dim_tabular: int = cfg.DIM_TABULAR,
        dim_fusao: int = cfg.DIM_FUSAO,
        dropout_fusao: float = cfg.DROPOUT_FUSAO,
        dropout_tabular: float = cfg.DROPOUT_TABULAR,
    ) -> None:
        super().__init__()
        self.branch_visual = _BranchVisual(pretrained=pretrained, dim_saida=dim_visual)
        self.branch_tabular = _BranchTabular(num_features_tabulares, dim_tabular, dropout_tabular)
        self.fusao = nn.Sequential(
            nn.Linear(dim_visual + dim_tabular, dim_fusao),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_fusao),
            nn.Linear(dim_fusao, num_classes),
        )

    def forward(self, imagem: torch.Tensor, tabular: torch.Tensor) -> torch.Tensor:
        feat_visual = self.branch_visual(imagem)
        feat_tabular = self.branch_tabular(tabular)
        fusao = torch.cat([feat_visual, feat_tabular], dim=1)
        return self.fusao(fusao)


class MultimodalGalaxy(ClassificadorGalaxias):
    """Wrapper ClassificadorGalaxias para RedeMultimodal."""

    def __init__(
        self,
        num_features_tabulares: int = len(cfg.FEATURES_TABULARES),
        pretrained: bool = True,
    ) -> None:
        self.num_features_tabulares = num_features_tabulares
        self.pretrained = pretrained

    @property
    def nome(self) -> str:
        return "Multimodal"

    @property
    def variante(self) -> str:
        return "cnn_mlp_fusion"

    @property
    def metodo_xai(self) -> str:
        return "grad-cam+shap"

    @property
    def camadas_xai(self) -> list[str]:
        return ["branch_visual.backbone.blocks.6.0.conv_pwl"]

    @property
    def suporta_finetune(self) -> bool:
        return True

    def construir(self, num_classes: int = 10, tamanho_imagem: int = 224) -> RedeMultimodal:
        return RedeMultimodal(
            num_classes=num_classes,
            num_features_tabulares=self.num_features_tabulares,
            pretrained=self.pretrained,
        )

    def explicar(
        self,
        rede: nn.Module,
        tensor_entrada: torch.Tensor,
        classe_alvo: Optional[int] = None,
    ) -> np.ndarray:
        from modelos.multimodal.xai import grad_cam_multimodal
        h, w = tensor_entrada.shape[2], tensor_entrada.shape[3]
        return grad_cam_multimodal(rede, tensor_entrada, classe_alvo, tamanho_saida=(h, w))
