"""Dataset e DataLoaders PyTorch para o projeto Galaxy Classification."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from pre_processamento.divisao_treino_teste import DivisaoDados


class DatasetGalaxias(Dataset):
    """Dataset PyTorch que envolve arrays numpy + transforms.

    Args:
        imagens: Array (N, H, W, C) uint8.
        rotulos: Array (N,) int64.
        transform: Pipeline de transforms torchvision.
    """

    def __init__(
        self,
        imagens: np.ndarray,
        rotulos: np.ndarray,
        transform: Optional[Callable] = None,
    ) -> None:
        self.imagens = imagens
        self.rotulos = rotulos.astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rotulos)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = self.imagens[idx]
        rot = int(self.rotulos[idx])
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, rot


def criar_dataloaders(
    divisao: DivisaoDados,
    transform_treino: transforms.Compose,
    transform_val: transforms.Compose,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Cria DataLoaders de treino, validação e teste.

    Args:
        divisao: Partições do dataset.
        transform_treino: Transforms com augmentation.
        transform_val: Transforms sem augmentation (val e teste).
        batch_size: Tamanho do batch.
        num_workers: Processos paralelos de carregamento.

    Returns:
        (loader_treino, loader_val, loader_teste)
    """
    usar_pin_memory = torch.cuda.is_available()
    usar_persistent = num_workers > 0

    ds_treino = DatasetGalaxias(divisao.imagens_treino, divisao.rotulos_treino, transform_treino)
    ds_val = DatasetGalaxias(divisao.imagens_val, divisao.rotulos_val, transform_val)
    ds_teste = DatasetGalaxias(divisao.imagens_teste, divisao.rotulos_teste, transform_val)

    loader_treino = DataLoader(
        ds_treino,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=usar_pin_memory,
        persistent_workers=usar_persistent,
        drop_last=True,
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=usar_pin_memory,
        persistent_workers=usar_persistent,
    )
    loader_teste = DataLoader(
        ds_teste,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=usar_pin_memory,
        persistent_workers=usar_persistent,
    )
    return loader_treino, loader_val, loader_teste
