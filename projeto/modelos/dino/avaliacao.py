"""Avaliação do DINO."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from modelos._avaliacao_padrao import avaliar_modelo
from modelos.dino import config as cfg
from modelos.dino.modelo import DinoGalaxy
from utils.metricas import ResultadoAvaliacao


def avaliar(caminho_pesos: Path, config_override: Optional[dict] = None) -> ResultadoAvaliacao:
    params = {
        "dataset": cfg.DATASET, "tamanho_imagem": cfg.TAMANHO_IMAGEM,
        "batch_size": cfg.BATCH_SIZE_AJUSTE_FINO, "seed": cfg.SEED,
        "num_workers": cfg.NUM_WORKERS, "nome_experimento": cfg.NOME_EXPERIMENTO,
        "modo": cfg.MODO_DINO, "backbone_hub": cfg.BACKBONE_DINO,
        "backbone_scratch": cfg.BACKBONE_SCRATCH,
    }
    if config_override:
        params.update(config_override)
    return avaliar_modelo(
        DinoGalaxy(modo=params["modo"], backbone_hub=params["backbone_hub"],
                   backbone_scratch=params["backbone_scratch"]),
        caminho_pesos, params["dataset"], params["tamanho_imagem"],
        params["batch_size"], params["seed"], params["num_workers"], params["nome_experimento"],
    )
