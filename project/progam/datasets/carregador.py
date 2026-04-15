"""
Carregador de datasets de galáxias.

Responsável por carregar dados brutos de H5 e preparar para processamento.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np

logger = logging.getLogger(__name__)


class CarregadorDataset:
    """Carregador de datasets SDSS e DECaLS em formato H5."""

    def __init__(self, caminho_datasets: Optional[Path] = None):
        """
        Inicializar carregador.

        Args:
            caminho_datasets: Caminho para diretório com datasets.
                            Se None, usa ../datasets relativo ao progam.
        """
        if caminho_datasets is None:
            caminho_datasets = Path(__file__).parent.parent.parent / "datasets"

        self.caminho_datasets = Path(caminho_datasets)
        self.logger = logger

    def carregar_sdss(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carregar dataset SDSS.

        Returns:
            Tupla (imagens, rótulos) com arrays numpy.

        Raises:
            FileNotFoundError: Se arquivo não encontrado.
            ValueError: Se arquivo H5 corrompido ou formato inválido.
        """
        caminho = self.caminho_datasets / "Galaxy10_SDSS.h5"

        if not caminho.exists():
            self.logger.error(f"Dataset SDSS não encontrado em {caminho}")
            raise FileNotFoundError(f"Dataset não encontrado: {caminho}")

        try:
            self.logger.info(f"Carregando dataset SDSS de {caminho}")
            with h5py.File(caminho, "r") as arquivo:
                imagens = np.array(arquivo["images"])
                rótulos = np.array(arquivo["ans"])

            self.logger.info(f"SDSS carregado: {imagens.shape} imagens, {rótulos.shape} rótulos")
            return imagens, rótulos

        except Exception as e:
            self.logger.error(f"Erro ao carregar SDSS: {e}")
            raise ValueError(f"Erro ao carregar dataset SDSS: {e}")

    def carregar_decals(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Carregar dataset DECaLS.

        Returns:
            Tupla (imagens, rótulos) com arrays numpy.

        Raises:
            FileNotFoundError: Se arquivo não encontrado.
            ValueError: Se arquivo H5 corrompido ou formato inválido.
        """
        caminho = self.caminho_datasets / "Galaxy10_DECals.h5"

        if not caminho.exists():
            self.logger.error(f"Dataset DECaLS não encontrado em {caminho}")
            raise FileNotFoundError(f"Dataset não encontrado: {caminho}")

        try:
            self.logger.info(f"Carregando dataset DECaLS de {caminho}")
            with h5py.File(caminho, "r") as arquivo:
                imagens = np.array(arquivo["images"])
                rótulos = np.array(arquivo["ans"])

            self.logger.info(f"DECaLS carregado: {imagens.shape} imagens, {rótulos.shape} rótulos")
            return imagens, rótulos

        except Exception as e:
            self.logger.error(f"Erro ao carregar DECaLS: {e}")
            raise ValueError(f"Erro ao carregar dataset DECaLS: {e}")

    def carregar_ambos(self) -> dict:
        """
        Carregar ambos datasets.

        Returns:
            Dicionário com chaves 'sdss' e 'decals', cada um contendo (imagens, rótulos).
        """
        resultado = {}

        try:
            resultado["sdss"] = self.carregar_sdss()
        except (FileNotFoundError, ValueError) as e:
            self.logger.warning(f"Não foi possível carregar SDSS: {e}")
            resultado["sdss"] = None

        try:
            resultado["decals"] = self.carregar_decals()
        except (FileNotFoundError, ValueError) as e:
            self.logger.warning(f"Não foi possível carregar DECaLS: {e}")
            resultado["decals"] = None

        return resultado
