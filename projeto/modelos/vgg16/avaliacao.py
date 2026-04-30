"""Avaliação do VGG16."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from modelos._avaliacao_padrao import avaliar_modelo
from modelos.vgg16 import config as cfg
from modelos.vgg16.modelo import VGG16Galaxy
from utils.metricas import ResultadoAvaliacao


def avaliar(caminho_pesos: Path, config_override: Optional[dict] = None) -> ResultadoAvaliacao:
    params = {
        "dataset": cfg.DATASET, "tamanho_imagem": cfg.TAMANHO_IMAGEM,
        "batch_size": cfg.BATCH_SIZE, "seed": cfg.SEED, "num_workers": cfg.NUM_WORKERS,
        "nome_experimento": cfg.NOME_EXPERIMENTO, "pretrained": cfg.PRETRAINED,
    }
    if config_override:
        params.update(config_override)
    return avaliar_modelo(
        VGG16Galaxy(pretrained=params["pretrained"]),
        caminho_pesos, params["dataset"], params["tamanho_imagem"],
        params["batch_size"], params["seed"], params["num_workers"], params["nome_experimento"],
    )
