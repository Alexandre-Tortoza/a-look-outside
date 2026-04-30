"""Pipeline de treino do CNN Baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from dataset.carregador import CarregadorDataset, resolver_nome_dataset
from dataset.dataset_pytorch import criar_dataloaders
from modelos.cnn.modelo import CNNBaseline
from modelos.treinador import HistoricoTreino, TreinadorModelo
from pre_processamento.divisao_treino_teste import dividir_estratificado
from pre_processamento.normalizacao import obter_transform_avaliacao, obter_transform_treino
from utils.config_loader import carregar_config, obter_config_modelo, obter_config_recursos
from utils.experimento import gerar_nome_experimento
from utils.logger import obter_logger
from utils.recursos import aplicar_batch_cap, obter_num_workers
from utils.reproducibilidade import fixar_semente


def treinar(config_override: Optional[dict[str, Any]] = None) -> HistoricoTreino:
    """Executa o treino completo do CNN Baseline.

    Args:
        config_override: Valores para sobrescrever o config.yaml.
    """
    config = carregar_config()
    params = obter_config_modelo("cnn", config)
    cfg_rec = obter_config_recursos(config)

    if config_override:
        params.update(config_override)

    # Defaults especificos que podem vir do override
    dataset = params.get("dataset", "decals")
    versao = params.get("versao_dataset", "raw")
    params.setdefault("versao_dataset", versao)

    num_workers = obter_num_workers(cfg_rec)
    batch_size = aplicar_batch_cap(params.get("batch_size", 32), cfg_rec)
    params["batch_size"] = batch_size

    nome_exp = params.get("nome_experimento") or gerar_nome_experimento(
        "cnn", dataset, versao, params["epocas"]
    )
    params["nome_experimento"] = nome_exp

    log = obter_logger(__name__, arquivo_log=Path("docs/cnn/treino.log"))
    log.info("Iniciando treino CNN | params: %s", params)

    fixar_semente(params["seed"])

    nome_dataset = resolver_nome_dataset(dataset, versao)
    imagens, rotulos = CarregadorDataset().carregar(nome_dataset)

    divisao = dividir_estratificado(imagens, rotulos, semente=params["seed"])
    log.info("Divisão: %s", divisao)

    transform_treino = obter_transform_treino(tamanho_imagem=params["tamanho_imagem"])
    transform_val = obter_transform_avaliacao(tamanho_imagem=params["tamanho_imagem"])

    loader_treino, loader_val, _ = criar_dataloaders(
        divisao, transform_treino, transform_val,
        batch_size=batch_size, num_workers=num_workers,
    )

    num_classes = len(set(rotulos.tolist()))
    rede = CNNBaseline().construir(num_classes=num_classes, tamanho_imagem=params["tamanho_imagem"])
    log.info("Modelo: %s | parâmetros: %d", rede.__class__.__name__,
             sum(p.numel() for p in rede.parameters()))

    treinador = TreinadorModelo(
        epocas=params["epocas"],
        lr=params["lr"],
        config_recursos=cfg_rec,
        paciencia_early_stop=params["paciencia_early_stop"],
        salvar_checkpoints=params.get("salvar_pesos", True),
        scheduler_ativo=params.get("scheduler_ativo", True),
        peso_decay=params.get("peso_decay", 1e-4),
        logger=log,
    )
    return treinador.treinar(rede, nome_exp, loader_treino, loader_val, params=params)


if __name__ == "__main__":
    historico = treinar()
    print(historico.resumo())
