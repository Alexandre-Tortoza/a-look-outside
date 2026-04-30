"""Pipeline de treino do VGG16."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from dataset.carregador import CarregadorDataset
from dataset.dataset_pytorch import criar_dataloaders
from modelos._transfer_learning import treinar_two_stage
from modelos.vgg16 import config as cfg
from modelos.vgg16.modelo import VGG16Galaxy
from modelos.treinador import HistoricoTreino
from pre_processamento.divisao_treino_teste import dividir_estratificado
from pre_processamento.normalizacao import obter_transform_avaliacao, obter_transform_treino
from utils.logger import obter_logger
from utils.reproducibilidade import fixar_semente


def treinar(config_override: Optional[dict] = None) -> HistoricoTreino:
    params = {
        "dataset": cfg.DATASET, "epocas": cfg.EPOCAS, "batch_size": cfg.BATCH_SIZE,
        "lr_cabeca": cfg.LR_CABECA, "lr_backbone": cfg.LR_BACKBONE,
        "epocas_congelado": cfg.EPOCAS_CONGELADO, "tamanho_imagem": cfg.TAMANHO_IMAGEM,
        "seed": cfg.SEED, "pretrained": cfg.PRETRAINED, "salvar_pesos": cfg.SALVAR_PESOS,
        "num_workers": cfg.NUM_WORKERS, "paciencia_early_stop": cfg.PACIENCIA_EARLY_STOP,
        "scheduler_ativo": cfg.SCHEDULER_ATIVO, "peso_decay": cfg.PESO_DECAY,
        "nome_experimento": cfg.NOME_EXPERIMENTO,
    }
    if config_override:
        params.update(config_override)

    log = obter_logger(__name__, arquivo_log=Path("docs/vgg16/treino.log"))
    fixar_semente(params["seed"])

    imagens, rotulos = CarregadorDataset().carregar(params["dataset"])
    divisao = dividir_estratificado(imagens, rotulos, semente=params["seed"])

    loader_treino, loader_val, _ = criar_dataloaders(
        divisao,
        obter_transform_treino(tamanho_imagem=params["tamanho_imagem"]),
        obter_transform_avaliacao(tamanho_imagem=params["tamanho_imagem"]),
        params["batch_size"], params["num_workers"],
    )

    num_classes = len(set(rotulos.tolist()))
    rede = VGG16Galaxy(pretrained=params["pretrained"]).construir(
        num_classes=num_classes, tamanho_imagem=params["tamanho_imagem"]
    )

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return treinar_two_stage(
        rede=rede, nome_experimento=params["nome_experimento"],
        loader_treino=loader_treino, loader_val=loader_val,
        epocas_total=params["epocas"], epocas_congelado=params["epocas_congelado"],
        lr_cabeca=params["lr_cabeca"], lr_backbone=params["lr_backbone"],
        paciencia=params["paciencia_early_stop"], salvar_checkpoints=params["salvar_pesos"],
        scheduler_ativo=params["scheduler_ativo"], peso_decay=params["peso_decay"],
        dispositivo=dispositivo, dir_pesos=Path("pesos"), dir_docs=Path("docs"), logger=log,
    )


if __name__ == "__main__":
    print(treinar().resumo())
