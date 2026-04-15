"""
Exemplo de uso da pipeline de balanceamento.

Demonstra como usar os balanceadores de forma programática.
"""

import logging
from pathlib import Path

import numpy as np

from datasets import CarregadorDataset, PipelineBalanceamento

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def exemplo_basico():
    """Exemplo básico: usar pipeline para processar todos os datasets."""
    logger.info("=" * 80)
    logger.info("EXEMPLO 1: Pipeline Completa (todos os datasets e técnicas)")
    logger.info("=" * 80)

    pipeline = PipelineBalanceamento(semente=42)

    # Executar com todas as técnicas padrão
    resultados = pipeline.executar(
        nome_dataset="ambos",
        balanceadores=["adasyn", "smote", "undersampling", "oversampling"],
        salvar=True,
    )

    logger.info(f"\nResultados: {resultados}")


def exemplo_tecnica_especifica():
    """Exemplo: aplicar apenas SMOTE a um dataset específico."""
    logger.info("\n" + "=" * 80)
    logger.info("EXEMPLO 2: Técnica Específica (apenas SMOTE em SDSS)")
    logger.info("=" * 80)

    pipeline = PipelineBalanceamento(semente=42)

    resultados = pipeline.executar(
        nome_dataset="sdss",
        balanceadores=["smote"],
        salvar=True,
    )


def exemplo_carregar_manualmente():
    """Exemplo: carregar dados manualmente e aplicar balanceador."""
    logger.info("\n" + "=" * 80)
    logger.info("EXEMPLO 3: Carregamento Manual de Dados")
    logger.info("=" * 80)

    # Carregar dados brutos
    carregador = CarregadorDataset()

    try:
        imagens_sdss, rótulos_sdss = carregador.carregar_sdss()
        logger.info(f"SDSS carregado: {imagens_sdss.shape} imagens")

        # Criar pipeline e obter balanceador específico
        pipeline = PipelineBalanceamento(semente=42)

        # Aplicar SMOTE
        bal_smote = pipeline.obter_balanceador("smote")
        imagens_smote, rótulos_smote = bal_smote.balancear(imagens_sdss, rótulos_sdss)

        logger.info(f"SMOTE aplicado: {imagens_smote.shape} imagens")

        # Aplicar Undersampling ao resultado
        bal_under = pipeline.obter_balanceador("undersampling")
        imagens_final, rótulos_final = bal_under.balancear(imagens_smote, rótulos_smote)

        logger.info(f"Undersampling aplicado: {imagens_final.shape} imagens")

    except FileNotFoundError as e:
        logger.warning(f"Dataset não encontrado: {e}")


def exemplo_hibrido():
    """Exemplo: usar balanceador híbrido com técnicas customizadas."""
    logger.info("\n" + "=" * 80)
    logger.info("EXEMPLO 4: Balanceador Híbrido Customizado")
    logger.info("=" * 80)

    from datasets.balanceadores import (
        BalanceadorHibrido,
        BalanceadorSMOTE,
        BalanceadorUndersampling,
    )

    pipeline = PipelineBalanceamento(semente=42)

    try:
        imagens_sdss, rótulos_sdss = pipeline.carregador.carregar_sdss()

        # Criar pipeline customizada: SMOTE + Undersampling
        bal_smote = BalanceadorSMOTE(semente=42)
        bal_under = BalanceadorUndersampling(semente=42)

        bal_hibrido = BalanceadorHibrido(
            balanceadores=[bal_smote, bal_under],
            semente=42,
        )

        imagens_bal, rótulos_bal = bal_hibrido.balancear(imagens_sdss, rótulos_sdss)

        logger.info(f"Pipeline híbrida aplicada: {imagens_bal.shape} imagens")

    except FileNotFoundError as e:
        logger.warning(f"Dataset não encontrado: {e}")


def exemplo_comparar_tecnicas():
    """Exemplo: comparar distribuições após diferentes técnicas."""
    logger.info("\n" + "=" * 80)
    logger.info("EXEMPLO 5: Comparar Distribuições")
    logger.info("=" * 80)

    pipeline = PipelineBalanceamento(semente=42)

    try:
        imagens, rótulos = pipeline.carregador.carregar_sdss()

        logger.info("\nDistribuição original:")
        dist_original = pipeline.carregador.carregador.obter_distribuicao_classes(rótulos)
        for classe, cont in sorted(dist_original.items()):
            logger.info(f"  Classe {classe}: {cont} amostras")

        # Testar diferentes técnicas
        tecnicas = ["smote", "undersampling", "oversampling"]

        for tecnica in tecnicas:
            bal = pipeline.obter_balanceador(tecnica)
            _, rótulos_bal = bal.balancear(imagens, rótulos)

            dist_bal = pipeline.carregador.carregador.obter_distribuicao_classes(
                rótulos_bal
            )
            logger.info(f"\n{tecnica.upper()}:")
            for classe, cont in sorted(dist_bal.items()):
                logger.info(f"  Classe {classe}: {cont} amostras")

    except FileNotFoundError as e:
        logger.warning(f"Dataset não encontrado: {e}")


def exemplo_salvar_carregar():
    """Exemplo: salvar e carregar datasets balanceados."""
    logger.info("\n" + "=" * 80)
    logger.info("EXEMPLO 6: Salvar e Carregar Datasets")
    logger.info("=" * 80)

    pipeline = PipelineBalanceamento(semente=42)
    caminho_saida = Path(__file__).parent / "dados_teste"

    try:
        imagens, rótulos = pipeline.carregador.carregar_sdss()

        # Aplicar balanceamento
        bal = pipeline.obter_balanceador("smote")
        imagens_bal, rótulos_bal = bal.balancear(imagens, rótulos)

        # Salvar
        caminho_arquivo = caminho_saida / "teste_smote.npz"
        pipeline.salvar_dataset(imagens_bal, rótulos_bal, caminho_arquivo)

        # Carregar
        dados = np.load(caminho_arquivo)
        imagens_carregadas = dados["imagens"]
        rótulos_carregados = dados["rótulos"]

        logger.info(f"Carregado: {imagens_carregadas.shape} imagens")

    except FileNotFoundError as e:
        logger.warning(f"Dataset não encontrado: {e}")


if __name__ == "__main__":
    try:
        # Descomentar o exemplo desejado

        # exemplo_basico()
        # exemplo_tecnica_especifica()
        # exemplo_carregar_manualmente()
        # exemplo_hibrido()
        exemplo_comparar_tecnicas()
        # exemplo_salvar_carregar()

    except Exception as e:
        logger.error(f"Erro: {e}", exc_info=True)
