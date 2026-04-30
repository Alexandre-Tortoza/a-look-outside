"""Fine-tuning supervisionado após pré-treino DINO."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from dataset.carregador import CarregadorDataset
from dataset.dataset_pytorch import criar_dataloaders
from modelos._transfer_learning import treinar_two_stage
from modelos.dino import config as cfg
from modelos.dino.modelo import DinoGalaxy, _ModeloDinoScratch
from modelos.treinador import HistoricoTreino
from pre_processamento.divisao_treino_teste import dividir_estratificado
from pre_processamento.normalizacao import obter_transform_avaliacao, obter_transform_treino
from utils.logger import obter_logger
from utils.reproducibilidade import fixar_semente


def ajustar_fino(
    caminho_pesos_pretreino: Optional[Path] = None,
    config_override: Optional[dict] = None,
) -> HistoricoTreino:
    """Fine-tuning supervisionado do backbone DINO.

    Se caminho_pesos_pretreino for None e MODO_DINO = "hub",
    usa o DINOv2 pré-treinado diretamente do torch.hub.

    Args:
        caminho_pesos_pretreino: Pesos do backbone pré-treinado (.pth).
                                 None para usar DINOv2 hub.
        config_override: Sobrescreve parâmetros do config.

    Returns:
        HistoricoTreino do fine-tuning.
    """
    params = {
        "dataset": cfg.DATASET, "seed": cfg.SEED,
        "epocas": cfg.EPOCAS_AJUSTE_FINO, "batch_size": cfg.BATCH_SIZE_AJUSTE_FINO,
        "lr_cabeca": cfg.LR_AJUSTE_FINO, "lr_backbone": cfg.LR_BACKBONE_AJUSTE,
        "epocas_congelado": cfg.EPOCAS_CONGELADO, "tamanho_imagem": cfg.TAMANHO_IMAGEM,
        "num_workers": cfg.NUM_WORKERS, "paciencia_early_stop": cfg.PACIENCIA_EARLY_STOP,
        "scheduler_ativo": cfg.SCHEDULER_ATIVO, "peso_decay": cfg.PESO_DECAY,
        "salvar_pesos": cfg.SALVAR_PESOS, "nome_experimento": cfg.NOME_EXPERIMENTO,
        "modo": cfg.MODO_DINO, "backbone_hub": cfg.BACKBONE_DINO,
        "backbone_scratch": cfg.BACKBONE_SCRATCH,
    }
    if config_override:
        params.update(config_override)

    log = obter_logger(__name__, arquivo_log=Path("docs/dino/ajuste_fino.log"))
    fixar_semente(params["seed"])

    imagens, rotulos = CarregadorDataset().carregar(params["dataset"])
    divisao = dividir_estratificado(imagens, rotulos, semente=params["seed"])
    num_classes = len(set(rotulos.tolist()))

    loader_treino, loader_val, _ = criar_dataloaders(
        divisao,
        obter_transform_treino(tamanho_imagem=params["tamanho_imagem"]),
        obter_transform_avaliacao(tamanho_imagem=params["tamanho_imagem"]),
        params["batch_size"], params["num_workers"],
    )

    classificador = DinoGalaxy(
        modo=params["modo"],
        backbone_hub=params["backbone_hub"],
        backbone_scratch=params["backbone_scratch"],
    )
    rede = classificador.construir(num_classes=num_classes, tamanho_imagem=params["tamanho_imagem"])

    # Carrega pesos do pré-treino se fornecidos
    if caminho_pesos_pretreino is not None and caminho_pesos_pretreino.exists():
        backbone = getattr(rede, "backbone", rede)
        backbone.load_state_dict(torch.load(caminho_pesos_pretreino, map_location="cpu"))
        log.info("Backbone carregado de %s", caminho_pesos_pretreino)

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
    historico = ajustar_fino()
    print(historico.resumo())
