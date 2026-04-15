"""
Gerenciador de datasets disponíveis.

Descobre, lista e gerencia datasets brutos e balanceados.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class TipoDataset(Enum):
    """Tipos de datasets disponíveis."""

    BRUTO = "bruto"
    BALANCEADO = "balanceado"


@dataclass
class InfoDataset:
    """Informações sobre um dataset."""

    nome: str
    tipo: TipoDataset
    caminho: Path
    tamanho_mb: float
    descricao: str
    num_amostras: Optional[int] = None
    num_classes: Optional[int] = None
    tecnica_balanceamento: Optional[str] = None


class GerenciadorDatasets:
    """Gerencia datasets disponíveis no projeto."""

    def __init__(self, caminho_datasets: Optional[Path] = None):
        """
        Inicializar gerenciador.

        Args:
            caminho_datasets: Caminho para diretório de datasets.
                            Se None, usa ../datasets relativo ao progam.
        """
        if caminho_datasets is None:
            caminho_datasets = Path(__file__).parent.parent.parent.parent / "datasets"

        self.caminho_datasets = Path(caminho_datasets)
        self.logger = logger

    def listar_datasets(self) -> List[InfoDataset]:
        """
        Listar todos os datasets disponíveis.

        Returns:
            Lista com informações de cada dataset.
        """
        datasets = []

        # Datasets brutos (H5)
        datasets.extend(self._descobrir_brutos())

        # Datasets balanceados (NPZ)
        datasets.extend(self._descobrir_balanceados())

        return sorted(datasets, key=lambda d: d.nome)

    def _descobrir_brutos(self) -> List[InfoDataset]:
        """Descobrir datasets brutos em formato H5."""
        datasets = []
        mapa_descricoes = {
            "Galaxy10_SDSS.h5": "SDSS - 21.785 galáxias de 10 classes",
            "Galaxy10_DECals.h5": "DECaLS - 8.671 galáxias de 10 classes",
        }

        if not self.caminho_datasets.exists():
            return datasets

        for arquivo_h5 in self.caminho_datasets.glob("Galaxy10_*.h5"):
            try:
                tamanho_mb = arquivo_h5.stat().st_size / (1024 ** 2)
                descricao = mapa_descricoes.get(arquivo_h5.name, "Dataset de galáxias")

                datasets.append(
                    InfoDataset(
                        nome=arquivo_h5.stem,
                        tipo=TipoDataset.BRUTO,
                        caminho=arquivo_h5,
                        tamanho_mb=tamanho_mb,
                        descricao=descricao,
                    )
                )
                self.logger.debug(f"Dataset bruto encontrado: {arquivo_h5.name}")

            except OSError as e:
                self.logger.warning(f"Erro ao ler {arquivo_h5.name}: {e}")

        return datasets

    def _descobrir_balanceados(self) -> List[InfoDataset]:
        """Descobrir datasets balanceados em formato NPZ."""
        datasets = []
        caminho_balanceados = self.caminho_datasets / "balanceados"

        if not caminho_balanceados.exists():
            return datasets

        for arquivo_npz in caminho_balanceados.glob("*.npz"):
            try:
                tamanho_mb = arquivo_npz.stat().st_size / (1024 ** 2)

                # Extrair nome do dataset e técnica
                nome_base = arquivo_npz.stem  # ex: "sdss_smote"
                partes = nome_base.split("_", 1)

                nome_dataset = partes[0].upper()
                tecnica = partes[1] if len(partes) > 1 else "desconhecido"

                descricao = f"{nome_dataset} - Balanceado com {tecnica.upper()}"

                datasets.append(
                    InfoDataset(
                        nome=nome_base,
                        tipo=TipoDataset.BALANCEADO,
                        caminho=arquivo_npz,
                        tamanho_mb=tamanho_mb,
                        descricao=descricao,
                        tecnica_balanceamento=tecnica,
                    )
                )
                self.logger.debug(f"Dataset balanceado encontrado: {arquivo_npz.name}")

            except OSError as e:
                self.logger.warning(f"Erro ao ler {arquivo_npz.name}: {e}")

        return datasets

    def pode_gerar_balanceado(self, nome_bruto: str) -> bool:
        """
        Verificar se um dataset bruto pode ser balanceado.

        Args:
            nome_bruto: Nome do dataset bruto (ex: "Galaxy10_SDSS").

        Returns:
            True se arquivo existe e pode ser processado.
        """
        caminho = self.caminho_datasets / f"{nome_bruto}.h5"
        return caminho.exists()

    def gerar_balanceado(
        self,
        nome_bruto: str,
        tecnicas: List[str],
    ) -> bool:
        """
        Gerar datasets balanceados a partir de um dataset bruto.

        Args:
            nome_bruto: Nome do dataset bruto.
            tecnicas: Lista de técnicas de balanceamento.

        Returns:
            True se geração bem-sucedida.
        """
        try:
            from datasets import PipelineBalanceamento

            # Mapear nomes de dataset bruto para nomes da pipeline
            nome_mapa = {
                "Galaxy10_SDSS": "sdss",
                "Galaxy10_DECals": "decals",
            }

            nome_pipeline = nome_mapa.get(nome_bruto)
            if not nome_pipeline:
                self.logger.error(f"Dataset {nome_bruto} não mapeado")
                return False

            pipeline = PipelineBalanceamento(
                caminho_datasets=self.caminho_datasets,
                semente=42,
            )

            resultados = pipeline.executar(
                nome_dataset=nome_pipeline,
                balanceadores=tecnicas,
                salvar=True,
            )

            self.logger.info(f"Datasets balanceados gerados com sucesso")
            return True

        except Exception as e:
            self.logger.error(f"Erro ao gerar dataset balanceado: {e}")
            return False
