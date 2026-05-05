"""DINO: modo hub (DINOv2 pré-treinado) ou pré-treino do zero."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from modelos.base import ClassificadorGalaxias
from modelos.dino.xai import mapas_atencao_dino


class _CabecaLinear(nn.Module):
    """Cabeça de classificação linear simples."""

    def __init__(self, dim_entrada: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim_entrada, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _CabecaMLP(nn.Module):
    """Cabeça de classificação MLP: Linear → GELU → Dropout → Linear."""

    def __init__(self, dim_entrada: int, num_classes: int, dropout: float = 0.1) -> None:
        super().__init__()
        dim_oculto = max(dim_entrada // 2, num_classes * 4)
        self.layers = nn.Sequential(
            nn.Linear(dim_entrada, dim_oculto),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_oculto, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _ModeloDinoHub(nn.Module):
    """DINOv2 via torch.hub + cabeça de classificação (linear ou MLP)."""

    def __init__(self, backbone_nome: str, num_classes: int, cabeca_mlp: bool = False) -> None:
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", backbone_nome, pretrained=True
        )
        dim = self.backbone.embed_dim
        self.head = _CabecaMLP(dim, num_classes) if cabeca_mlp else _CabecaLinear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


class _ModeloDinoScratch(nn.Module):
    """ViT pequeno + cabeça de projeção DINO + cabeça de classificação."""

    def __init__(self, backbone_nome: str, num_classes: int, tamanho_projecao: int = 65536) -> None:
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone_nome, pretrained=False, num_classes=0)
        dim = self.backbone.embed_dim
        self.projecao = CabecaProjecaoDINO(dim, tamanho_projecao)
        self.classificador = _CabecaLinear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classificador(features)

    def projetar(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna embeddings projetados — usado no pré-treino."""
        features = self.backbone(x)
        return self.projecao(features)


class CabecaProjecaoDINO(nn.Module):
    """MLP de 3 camadas para projeção no espaço de destilação."""

    def __init__(self, dim_entrada: int, dim_saida: int = 65536) -> None:
        super().__init__()
        dim_oculto = 2048
        self.layers = nn.Sequential(
            nn.Linear(dim_entrada, dim_oculto),
            nn.GELU(),
            nn.Linear(dim_oculto, dim_oculto),
            nn.GELU(),
            nn.Linear(dim_oculto, dim_saida, bias=False),
        )
        # Normalização L2 na saída
        self.norm = nn.functional.normalize if False else None  # aplicada no forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layers(x)
        return nn.functional.normalize(out, dim=-1)


class DinoGalaxy(ClassificadorGalaxias):
    """Wrapper DINO para ClassificadorGalaxias."""

    def __init__(
        self,
        modo: str = "hub",
        backbone_hub: str = "dinov2_vitb14",
        backbone_scratch: str = "vit_small_patch16_224",
        tamanho_projecao: int = 65536,
        cabeca_mlp: bool = False,
    ) -> None:
        self.modo = modo
        self.backbone_hub = backbone_hub
        self.backbone_scratch = backbone_scratch
        self.tamanho_projecao = tamanho_projecao
        self.cabeca_mlp = cabeca_mlp

    @property
    def nome(self) -> str:
        return "DINO"

    @property
    def variante(self) -> str:
        return f"self-supervised-{self.modo}"

    @property
    def metodo_xai(self) -> str:
        return "attention-maps"

    @property
    def suporta_finetune(self) -> bool:
        return True

    @property
    def suporta_checkpoint(self) -> bool:
        return True

    def construir(self, num_classes: int = 10, tamanho_imagem: int = 224) -> nn.Module:
        if self.modo == "hub":
            return _ModeloDinoHub(self.backbone_hub, num_classes, cabeca_mlp=self.cabeca_mlp)
        return _ModeloDinoScratch(self.backbone_scratch, num_classes, self.tamanho_projecao)

    def explicar(
        self,
        rede: nn.Module,
        tensor_entrada: torch.Tensor,
        classe_alvo: Optional[int] = None,
    ) -> np.ndarray:
        h, w = tensor_entrada.shape[2], tensor_entrada.shape[3]
        return mapas_atencao_dino(rede, tensor_entrada, tamanho_saida=(h, w))
