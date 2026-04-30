"""Pipeline de treino do CNN Baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from dataset.carregador import CarregadorDataset
from dataset.dataset_pytorch import criar_dataloaders
from modelos.cnn import config as cfg
from modelos.cnn.modelo import CNNBaseline
from modelos.treinador import HistoricoTreino, TreinadorModelo
from pre_processamento.divisao_treino_teste import dividir_estratificado
from pre_processamento.normalizacao import obter_transform_avaliacao, obter_transform_treino
from utils.logger import obter_logger
from utils.reproducibilidade import fixar_semente


def treinar(config_override: Optional[dict] = None) -> HistoricoTreino:
    """Executa o treino completo do CNN Baseline.

    Args:
        config_override: Dicionário com valores para sobrescrever o config.py.
                         Chaves: qualquer constante de modelos/cnn/config.py.

    Returns:
        HistoricoTreino com métricas de todas as épocas.
    """
    # Aplica overrides
    params = {
        "dataset": cfg.DATASET,
        "versao_dataset": cfg.VERSAO_DATASET,
        "epocas": cfg.EPOCAS,
        "batch_size": cfg.BATCH_SIZE,
        "lr": cfg.LR,
        "tamanho_imagem": cfg.TAMANHO_IMAGEM,
        "seed": cfg.SEED,
        "salvar_pesos": cfg.SALVAR_PESOS,
        "num_workers": cfg.NUM_WORKERS,
        "paciencia_early_stop": cfg.PACIENCIA_EARLY_STOP,
        "scheduler_ativo": cfg.SCHEDULER_ATIVO,
        "peso_decay": cfg.PESO_DECAY,
        "nome_experimento": cfg.NOME_EXPERIMENTO,
    }
    if config_override:
        params.update(config_override)

    log = obter_logger(
        __name__,
        arquivo_log=Path("docs") / params["nome_experimento"].split("_")[0] / "treino.log",
    )
    log.info("Iniciando treino CNN | params: %s", params)

    # 1. Reprodutibilidade
    fixar_semente(params["seed"])

    # 2. Carregar dataset
    carregador = CarregadorDataset()
    imagens, rotulos = carregador.carregar(params["dataset"])

    # 3. Divisão estratificada
    divisao = dividir_estratificado(imagens, rotulos, semente=params["seed"])
    log.info("Divisão: %s", divisao)

    # 4. Transforms
    transform_treino = obter_transform_treino(tamanho_imagem=params["tamanho_imagem"])
    transform_val = obter_transform_avaliacao(tamanho_imagem=params["tamanho_imagem"])

    # 5. DataLoaders
    loader_treino, loader_val, _ = criar_dataloaders(
        divisao,
        transform_treino,
        transform_val,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
    )

    # 6. Modelo
    num_classes = len(set(rotulos.tolist()))
    rede = CNNBaseline().construir(num_classes=num_classes, tamanho_imagem=params["tamanho_imagem"])
    log.info("Modelo: %s | parâmetros: %d", rede.__class__.__name__,
             sum(p.numel() for p in rede.parameters()))

    # 7. Treinar
    treinador = TreinadorModelo(
        epocas=params["epocas"],
        lr=params["lr"],
        paciencia_early_stop=params["paciencia_early_stop"],
        salvar_checkpoints=params["salvar_pesos"],
        scheduler_ativo=params["scheduler_ativo"],
        peso_decay=params["peso_decay"],
        logger=log,
    )
    return treinador.treinar(rede, params["nome_experimento"], loader_treino, loader_val)


if __name__ == "__main__":
    historico = treinar()
    print(historico.resumo())
