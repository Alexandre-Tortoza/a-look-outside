"""Preparador de dados: converte H5/NPZ em DataLoaders PyTorch."""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import h5py


class PreparadorDados:
    """Carrega dados e cria DataLoaders com split treino/validação."""

    def preparar(
        self,
        caminho_dataset: str,
        batch_size: int,
        num_workers: int,
        divisao_treino: float,
        img_size: int,
    ) -> tuple:
        """
        Carrega dataset e retorna DataLoaders de treino e validação.

        Args:
            caminho_dataset: Caminho para arquivo H5 ou NPZ
            batch_size: Tamanho do batch
            num_workers: Número de workers para DataLoader
            divisao_treino: Fração para treino (ex: 0.7 = 70% treino, 30% val)
            img_size: Tamanho da imagem (redimensiona para img_size × img_size)

        Returns:
            (loader_treino, loader_val): DataLoaders de treino e validação
        """
        caminho = Path(caminho_dataset)

        if caminho.suffix == ".h5":
            imagens, rotulos = self._carregar_h5(caminho)
        elif caminho.suffix == ".npz":
            imagens, rotulos = self._carregar_npz(caminho)
        else:
            raise ValueError(f"Formato não suportado: {caminho.suffix}")

        # Dividir em treino/val
        num_samples = len(imagens)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        split_idx = int(num_samples * divisao_treino)
        indices_treino = indices[:split_idx]
        indices_val = indices[split_idx:]

        imagens_treino = imagens[indices_treino]
        rotulos_treino = rotulos[indices_treino]
        imagens_val = imagens[indices_val]
        rotulos_val = rotulos[indices_val]

        # Transformações
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Criar datasets
        dataset_treino = _DatasetComTransform(imagens_treino, rotulos_treino, transform)
        dataset_val = _DatasetComTransform(imagens_val, rotulos_val, transform)

        # Criar DataLoaders
        loader_treino = DataLoader(
            dataset_treino,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        loader_val = DataLoader(
            dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        return loader_treino, loader_val

    @staticmethod
    def _carregar_h5(caminho: Path) -> tuple:
        """Carrega imagens e rótulos de arquivo H5."""
        with h5py.File(caminho, "r") as f:
            imagens = f["images"][:]
            rotulos = f["ans"][:]
        return imagens, rotulos.astype(np.int64)

    @staticmethod
    def _carregar_npz(caminho: Path) -> tuple:
        """Carrega imagens e rótulos de arquivo NPZ."""
        dados = np.load(caminho)
        imagens = dados["imagens"]
        rotulos = dados["rótulos"].astype(np.int64)
        return imagens, rotulos


class _DatasetComTransform(torch.utils.data.Dataset):
    """Dataset wrapper que aplica transformações."""

    def __init__(self, imagens: np.ndarray, rotulos: np.ndarray, transform=None):
        self.imagens = imagens
        self.rotulos = rotulos
        self.transform = transform

    def __len__(self):
        return len(self.imagens)

    def __getitem__(self, idx):
        img = self.imagens[idx]
        label = self.rotulos[idx]

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).float()

        return img, label
