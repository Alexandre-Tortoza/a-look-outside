"""
Pipeline de balanceamento de datasets.

Orquestra carregamento de dados, aplicação de balanceadores e salvamento de resultados.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .balanceadores import (
    BalanceadorADASYN,
    BalanceadorEstratificacao,
    BalanceadorHibrido,
    BalanceadorOversampling,
    BalanceadorSMOTE,
    BalanceadorUndersampling,
)
from .carregador import CarregadorDataset

logger = logging.getLogger(__name__)

# Mapa de balanceadores disponíveis
BALANCEADORES_DISPONIVEIS = {
    "adasyn": BalanceadorADASYN,
    "smote": BalanceadorSMOTE,
    "undersampling": BalanceadorUndersampling,
    "oversampling": BalanceadorOversampling,
    "estratificacao": BalanceadorEstratificacao,
    "hibrido": BalanceadorHibrido,
}


class PipelineBalanceamento:
    """Pipeline para balanceamento de datasets de galáxias."""

    def __init__(
        self,
        caminho_datasets: Optional[Path] = None,
        caminho_saida: Optional[Path] = None,
        semente: Optional[int] = None,
    ):
        """
        Inicializar pipeline.

        Args:
            caminho_datasets: Caminho para diretório com datasets brutos.
            caminho_saida: Caminho para salvar datasets balanceados.
            semente: Seed para reprodutibilidade.
        """
        self.carregador = CarregadorDataset(caminho_datasets)

        if caminho_saida is None:
            caminho_saida = Path(__file__).parent.parent.parent / "datasets" / "balanceados"

        self.caminho_saida = Path(caminho_saida)
        self.semente = semente
        self.logger = logger

    def obter_balanceador(self, nome: str, **kwargs) -> object:
        """
        Obter instância de balanceador pelo nome.

        Args:
            nome: Nome do balanceador.
            **kwargs: Argumentos adicionais para o balanceador.

        Returns:
            Instância do balanceador.

        Raises:
            ValueError: Se balanceador não existe.
        """
        nome = nome.lower().strip()

        if nome not in BALANCEADORES_DISPONIVEIS:
            raise ValueError(
                f"Balanceador '{nome}' não encontrado. "
                f"Disponíveis: {list(BALANCEADORES_DISPONIVEIS.keys())}"
            )

        classe_balanceador = BALANCEADORES_DISPONIVEIS[nome]
        return classe_balanceador(semente=self.semente, **kwargs)

    def listar_balanceadores(self) -> List[str]:
        """
        Listar balanceadores disponíveis.

        Returns:
            Lista com nomes dos balanceadores.
        """
        return list(BALANCEADORES_DISPONIVEIS.keys())

    def processar_dataset(
        self,
        imagens: np.ndarray,
        rótulos: np.ndarray,
        nome_dataset: str,
        balanceadores: List[str],
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Aplicar balanceadores a um dataset.

        Args:
            imagens: Array com imagens.
            rótulos: Array com rótulos.
            nome_dataset: Nome do dataset (para logging).
            balanceadores: Lista com nomes dos balanceadores a aplicar.

        Returns:
            Dicionário {nome_balanceador: (imagens_balanceadas, rótulos_balanceados)}.
        """
        self.logger.info(f"\n{'=' * 60}")
        self.logger.info(f"Processando dataset: {nome_dataset}")
        self.logger.info(f"{'=' * 60}")

        resultados = {}

        for nome_balanceador in balanceadores:
            try:
                self.logger.info(f"\nAplicando {nome_balanceador}...")
                balanceador = self.obter_balanceador(nome_balanceador)

                imagens_bal, rótulos_bal = balanceador.balancear(imagens, rótulos)
                resultados[nome_balanceador] = (imagens_bal, rótulos_bal)

                self.logger.info(
                    f"✓ {nome_balanceador} concluído: "
                    f"{len(rótulos)} → {len(rótulos_bal)} amostras"
                )

            except Exception as e:
                self.logger.error(f"✗ Erro em {nome_balanceador}: {e}")
                resultados[nome_balanceador] = None

        return resultados

    def salvar_dataset(
        self,
        imagens: np.ndarray,
        rótulos: np.ndarray,
        caminho: Path,
    ) -> None:
        """
        Salvar dataset em formato NPZ.

        Args:
            imagens: Array com imagens.
            rótulos: Array com rótulos.
            caminho: Caminho para salvar arquivo.
        """
        caminho.parent.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Salvando em {caminho}...")
        np.savez_compressed(
            caminho,
            imagens=imagens,
            rótulos=rótulos,
        )
        self.logger.info(f"✓ Dataset salvo: {caminho}")

    def executar(
        self,
        nome_dataset: str = "ambos",
        balanceadores: Optional[List[str]] = None,
        salvar: bool = True,
    ) -> Dict:
        """
        Executar pipeline completa.

        Args:
            nome_dataset: 'sdss', 'decals', ou 'ambos'.
            balanceadores: Lista de balanceadores a aplicar.
                         Se None, aplica todos os disponíveis.
            salvar: Se True, salva datasets processados.

        Returns:
            Dicionário com resultados por dataset e balanceador.
        """
        if balanceadores is None:
            balanceadores = self.listar_balanceadores()

        self.logger.info(f"Iniciando pipeline de balanceamento")
        self.logger.info(f"Datasets: {nome_dataset}")
        self.logger.info(f"Técnicas: {balanceadores}")

        resultados_finais = {}

        # Carregar datasets
        if nome_dataset in ["sdss", "ambos"]:
            try:
                self.logger.info("\nCarregando SDSS...")
                imagens_sdss, rótulos_sdss = self.carregador.carregar_sdss()

                resultados = self.processar_dataset(
                    imagens_sdss,
                    rótulos_sdss,
                    "SDSS",
                    balanceadores,
                )

                if salvar:
                    self.caminho_saida.mkdir(parents=True, exist_ok=True)
                    for nome_bal, dados in resultados.items():
                        if dados is not None:
                            imagens_bal, rótulos_bal = dados
                            caminho = self.caminho_saida / f"sdss_{nome_bal}.npz"
                            self.salvar_dataset(imagens_bal, rótulos_bal, caminho)

                resultados_finais["sdss"] = resultados

            except Exception as e:
                self.logger.error(f"Erro ao processar SDSS: {e}")
                resultados_finais["sdss"] = None

        if nome_dataset in ["decals", "ambos"]:
            try:
                self.logger.info("\nCarregando DECaLS...")
                imagens_decals, rótulos_decals = self.carregador.carregar_decals()

                resultados = self.processar_dataset(
                    imagens_decals,
                    rótulos_decals,
                    "DECaLS",
                    balanceadores,
                )

                if salvar:
                    self.caminho_saida.mkdir(parents=True, exist_ok=True)
                    for nome_bal, dados in resultados.items():
                        if dados is not None:
                            imagens_bal, rótulos_bal = dados
                            caminho = self.caminho_saida / f"decals_{nome_bal}.npz"
                            self.salvar_dataset(imagens_bal, rótulos_bal, caminho)

                resultados_finais["decals"] = resultados

            except Exception as e:
                self.logger.error(f"Erro ao processar DECaLS: {e}")
                resultados_finais["decals"] = None

        self.logger.info(f"\n{'=' * 60}")
        self.logger.info("Pipeline concluída!")
        self.logger.info(f"{'=' * 60}")

        return resultados_finais
