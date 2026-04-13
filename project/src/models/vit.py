"""Arquiteturas Vision Transformer para classificação de morfologia galáctica."""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from models.base import GalaxyClassifier


class _IncorporadorPatch(nn.Module):
    """Converte imagem em sequência de patches embedados.

    Usa convolução com kernel=patch_size e stride=patch_size
    para extrair patches não-sobrepostos de forma eficiente.
    """

    def __init__(self, tamanho_imagem: int, tamanho_patch: int, dimensao_embed: int):
        super().__init__()
        self.proj = nn.Conv2d(3, dimensao_embed, kernel_size=tamanho_patch, stride=tamanho_patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna tensor de forma (B, num_patches, dimensao_embed)."""
        x = self.proj(x)  # (B, dimensao_embed, H', W')
        return x.flatten(2).transpose(1, 2)  # (B, H'*W', dimensao_embed)


class _BlocoTransformador(nn.Module):
    """Bloco transformer com atenção multi-cabeça e MLP.

    Usa arquitetura pre-norm: LayerNorm antes de cada operação.
    """

    def __init__(self, dimensao_embed: int, num_cabecas: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dimensao_embed)
        self.atencao = nn.MultiheadAttention(
            dimensao_embed, num_cabecas, dropout=0.1, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dimensao_embed)
        self.rede_densa = nn.Sequential(
            nn.Linear(dimensao_embed, 4 * dimensao_embed),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4 * dimensao_embed, dimensao_embed),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        atencao_saida, _ = self.atencao(x_norm, x_norm, x_norm)
        x = x + atencao_saida

        x = x + self.rede_densa(self.norm2(x))
        return x


class RedeViTSimples(nn.Module):
    """Vision Transformer leve para classificação de galáxias.

    Arquitetura:
    - Patches de 16×16, dimensão de embedding 192
    - 6 blocos transformer, 4 cabeças de atenção
    - Token CLS para classificação
    - Embeddings de posição treináveis

    Otimizada para iteração rápida.
    """

    def __init__(
        self,
        tamanho_imagem: int,
        tamanho_patch: int,
        dimensao_embed: int,
        num_cabecas: int,
        num_blocos: int,
        num_classes: int,
    ):
        super().__init__()
        assert tamanho_imagem % tamanho_patch == 0, (
            f"tamanho_imagem={tamanho_imagem} deve ser divisível por "
            f"tamanho_patch={tamanho_patch}. Para SDSS (69px), redimensione para 64px."
        )

        num_patches = (tamanho_imagem // tamanho_patch) ** 2

        self.incorporador_patch = _IncorporadorPatch(tamanho_imagem, tamanho_patch, dimensao_embed)
        self.token_cls = nn.Parameter(torch.zeros(1, 1, dimensao_embed))
        self.embedding_posicao = nn.Parameter(torch.zeros(1, num_patches + 1, dimensao_embed))

        nn.init.trunc_normal_(self.token_cls, std=0.02)
        nn.init.trunc_normal_(self.embedding_posicao, std=0.02)

        self.blocos = nn.Sequential(
            *[_BlocoTransformador(dimensao_embed, num_cabecas) for _ in range(num_blocos)]
        )
        self.norm = nn.LayerNorm(dimensao_embed)
        self.cabeca = nn.Linear(dimensao_embed, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.incorporador_patch(x)

        # Prepend CLS token
        token_cls = self.token_cls.expand(B, -1, -1)
        x = torch.cat([token_cls, x], dim=1)

        # Adiciona embeddings de posição
        x = x + self.embedding_posicao

        # Aplica blocos transformer
        x = self.blocos(x)

        # Normaliza e extrai token CLS
        x = self.norm(x)
        x = x[:, 0]

        return self.cabeca(x)


class RedeViTRobusta(nn.Module):
    """Vision Transformer robusta para classificação de galáxias.

    Arquitetura:
    - Patches de 8×8, dimensão de embedding 768
    - 12 blocos transformer, 12 cabeças de atenção (multi-escala)
    - Token CLS para classificação
    - Embeddings de posição treináveis
    - Cabeça de 2 camadas para classificação

    Otimizada para máxima generalização cross-dataset.
    """

    def __init__(
        self,
        tamanho_imagem: int,
        tamanho_patch: int,
        dimensao_embed: int,
        num_cabecas: int,
        num_blocos: int,
        num_classes: int,
    ):
        super().__init__()
        assert tamanho_imagem % tamanho_patch == 0, (
            f"tamanho_imagem={tamanho_imagem} deve ser divisível por "
            f"tamanho_patch={tamanho_patch}. Para SDSS (69px), redimensione para 64px."
        )

        num_patches = (tamanho_imagem // tamanho_patch) ** 2

        self.incorporador_patch = _IncorporadorPatch(tamanho_imagem, tamanho_patch, dimensao_embed)
        self.token_cls = nn.Parameter(torch.zeros(1, 1, dimensao_embed))
        self.embedding_posicao = nn.Parameter(torch.zeros(1, num_patches + 1, dimensao_embed))

        nn.init.trunc_normal_(self.token_cls, std=0.02)
        nn.init.trunc_normal_(self.embedding_posicao, std=0.02)

        self.blocos = nn.Sequential(
            *[_BlocoTransformador(dimensao_embed, num_cabecas) for _ in range(num_blocos)]
        )
        self.norm = nn.LayerNorm(dimensao_embed)
        self.cabeca = nn.Sequential(
            nn.Linear(dimensao_embed, dimensao_embed),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dimensao_embed, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.incorporador_patch(x)

        # Prepend CLS token
        token_cls = self.token_cls.expand(B, -1, -1)
        x = torch.cat([token_cls, x], dim=1)

        # Adiciona embeddings de posição
        x = x + self.embedding_posicao

        # Aplica blocos transformer
        x = self.blocos(x)

        # Normaliza e extrai token CLS
        x = self.norm(x)
        x = x[:, 0]

        return self.cabeca(x)


class ViTLight(GalaxyClassifier):
    """Wrapper Vision Transformer Simples para orquestração de benchmark."""

    @property
    def name(self) -> str:
        return "ViT"

    @property
    def variant(self) -> str:
        return "light"

    @property
    def xai_method(self) -> str:
        return "attention-rollout"

    def build(self, num_classes: int, img_size: int) -> nn.Module:
        """Instancia a rede ViT simples."""
        return RedeViTSimples(
            tamanho_imagem=img_size,
            tamanho_patch=16,
            dimensao_embed=192,
            num_cabecas=4,
            num_blocos=6,
            num_classes=num_classes,
        )

    def explain(self, model, input_tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Retorna mapa Attention Rollout (stub)."""
        h, w = input_tensor.shape[-2], input_tensor.shape[-1]
        return np.zeros((h, w), dtype=np.float32)


class ViTRobust(GalaxyClassifier):
    """Wrapper Vision Transformer Robusta para orquestração de benchmark."""

    @property
    def name(self) -> str:
        return "ViT"

    @property
    def variant(self) -> str:
        return "robust"

    @property
    def xai_method(self) -> str:
        return "attention-rollout"

    def build(self, num_classes: int, img_size: int) -> nn.Module:
        """Instancia a rede ViT robusta."""
        return RedeViTRobusta(
            tamanho_imagem=img_size,
            tamanho_patch=8,
            dimensao_embed=768,
            num_cabecas=12,
            num_blocos=12,
            num_classes=num_classes,
        )

    def explain(self, model, input_tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Retorna mapa Attention Rollout (stub)."""
        h, w = input_tensor.shape[-2], input_tensor.shape[-1]
        return np.zeros((h, w), dtype=np.float32)
