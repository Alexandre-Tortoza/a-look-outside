"""Carregamento de datasets Galaxy10 (SDSS e DECaLS) de arquivos H5/NPZ."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import h5py
import numpy as np

_log = logging.getLogger(__name__)

# Mapeamento de nomes lógicos para subpastas dentro de dataset/raw/
_NOMES_RAW = {
    "sdss": "raw/sdss/galaxy10_sdss.h5",
    "decals": "raw/decals/galaxy10_decals.h5",
}


class CarregadorDataset:
    """Carrega datasets Galaxy10 a partir de arquivos H5 ou NPZ.

    Args:
        raiz: Diretório raiz da pasta dataset/ (padrão: dataset/ relativo ao CWD).
    """

    def __init__(self, raiz: Union[str, Path] = Path("dataset")) -> None:
        self.raiz = Path(raiz)

    def _resolver_caminho(self, nome: str) -> Path:
        """Resolve o nome lógico para um caminho de arquivo concreto."""
        if nome.endswith(".h5") or nome.endswith(".npz"):
            return Path(nome)
        if nome in _NOMES_RAW:
            return self.raiz / _NOMES_RAW[nome]
        # Tenta como caminho relativo à raiz
        for ext in (".h5", ".npz"):
            candidato = self.raiz / "processados" / (nome + ext)
            if candidato.exists():
                return candidato
        raise FileNotFoundError(
            f"Dataset '{nome}' não encontrado. "
            f"Coloque o arquivo em {self.raiz / _NOMES_RAW.get(nome, 'dataset/raw/')} "
            f"ou passe o caminho completo."
        )

    def carregar(self, nome: str) -> tuple[np.ndarray, np.ndarray]:
        """Carrega imagens e rótulos do dataset.

        Args:
            nome: "sdss", "decals", caminho .h5 ou caminho .npz.

        Returns:
            (imagens, rotulos) — arrays numpy.
            imagens: uint8 (N, H, W, 3) ou (N, H, W).
            rotulos: int64 (N,).
        """
        caminho = self._resolver_caminho(nome)
        tamanho_mb = caminho.stat().st_size / 1e6 if caminho.exists() else 0
        _log.info("Carregando '%s' (%.1f MB)…", caminho, tamanho_mb)

        if str(caminho).endswith(".npz"):
            return self._carregar_npz(caminho)
        return self._carregar_h5(caminho)

    def _carregar_h5(self, caminho: Path) -> tuple[np.ndarray, np.ndarray]:
        with h5py.File(caminho, "r") as f:
            imagens = f["images"][:]
            rotulos = f["ans"][:].astype(np.int64)
        _log.info("Carregado: %d amostras, shape imagens %s.", len(rotulos), imagens.shape)
        return imagens, rotulos

    def _carregar_npz(self, caminho: Path) -> tuple[np.ndarray, np.ndarray]:
        dados = np.load(caminho)
        # Suporta chaves 'images'/'ans' (Galaxy10) ou 'imagens'/'rotulos' (nossa convenção)
        if "images" in dados:
            imagens, rotulos = dados["images"], dados["ans"].astype(np.int64)
        else:
            imagens, rotulos = dados["imagens"], dados["rotulos"].astype(np.int64)
        _log.info("Carregado: %d amostras, shape imagens %s.", len(rotulos), imagens.shape)
        return imagens, rotulos

    def inspecionar(self, nome: str) -> dict:
        """Retorna metadados sem carregar todos os dados na memória.

        Returns:
            Dicionário com shape, dtype e distribuição de classes.
        """
        caminho = self._resolver_caminho(nome)
        with h5py.File(caminho, "r") as f:
            shape = tuple(f["images"].shape)
            dtype = str(f["images"].dtype)
            rotulos = f["ans"][:].astype(np.int64)

        contagens = np.bincount(rotulos).tolist()
        return {
            "caminho": str(caminho),
            "n_amostras": shape[0],
            "shape_imagem": shape[1:],
            "dtype": dtype,
            "n_classes": len(contagens),
            "contagem_por_classe": contagens,
        }
