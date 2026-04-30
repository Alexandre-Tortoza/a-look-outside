"""Pré-treino self-supervised do DINO (destilação professor-aluno).

Só relevante quando config.MODO_DINO = "pre_treino".
Para a maioria dos experimentos, prefira o modo "hub" (DINOv2 pré-treinado).
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.carregador import CarregadorDataset
from dataset.dataset_pytorch import criar_dataloaders
from modelos.dino import config as cfg
from modelos.dino.modelo import DinoGalaxy, _ModeloDinoScratch
from pre_processamento.divisao_treino_teste import dividir_estratificado
from pre_processamento.normalizacao import obter_transform_treino
from utils.logger import obter_logger
from utils.reproducibilidade import fixar_semente


def pre_treinar(config_override: Optional[dict] = None) -> Path:
    """Executa o pré-treino DINO self-supervised.

    Retorna o caminho para os pesos salvos do professor/backbone.

    Teacher: EMA (exponential moving average) dos pesos do aluno.
    Loss: cross-entropy entre distribuições suavizadas pelo temperature.
    """
    params = {
        "dataset": cfg.DATASET, "seed": cfg.SEED,
        "epocas": cfg.EPOCAS_PRE_TREINO, "batch_size": cfg.BATCH_SIZE_PRE_TREINO,
        "lr": cfg.LR_PRE_TREINO, "warmup_epocas": cfg.WARMUP_EPOCAS,
        "tamanho_projecao": cfg.TAMANHO_PROJECAO,
        "temperatura_professor": cfg.TEMPERATURA_PROFESSOR,
        "temperatura_aluno": cfg.TEMPERATURA_ALUNO,
        "backbone": cfg.BACKBONE_SCRATCH,
        "num_workers": cfg.NUM_WORKERS, "tamanho_imagem": cfg.TAMANHO_IMAGEM,
        "nome_experimento": cfg.NOME_EXPERIMENTO_PRE_TREINO,
    }
    if config_override:
        params.update(config_override)

    log = obter_logger(__name__, arquivo_log=Path("docs/dino/pre_treino.log"))
    log.info("Iniciando pré-treino DINO | épocas=%d", params["epocas"])
    fixar_semente(params["seed"])

    imagens, rotulos = CarregadorDataset().carregar(params["dataset"])
    divisao = dividir_estratificado(imagens, rotulos, semente=params["seed"])

    loader_treino, _, _ = criar_dataloaders(
        divisao,
        obter_transform_treino(tamanho_imagem=params["tamanho_imagem"]),
        obter_transform_treino(tamanho_imagem=params["tamanho_imagem"], aumentar=False),
        params["batch_size"], params["num_workers"],
    )

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = len(set(rotulos.tolist()))

    # Aluno e professor com mesma arquitetura
    aluno: _ModeloDinoScratch = DinoGalaxy(
        modo="pre_treino", backbone_scratch=params["backbone"],
        tamanho_projecao=params["tamanho_projecao"]
    ).construir(num_classes=num_classes, tamanho_imagem=params["tamanho_imagem"])
    aluno = aluno.to(dispositivo)

    professor = copy.deepcopy(aluno)
    for param in professor.parameters():
        param.requires_grad = False

    otimizador = torch.optim.AdamW(aluno.parameters(), lr=params["lr"], weight_decay=1e-4)
    ema_momentum = 0.996

    caminho_saida = Path("pesos/dino") / f"{params['nome_experimento']}.pth"
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    for epoca in range(1, params["epocas"] + 1):
        # LR warmup linear
        if epoca <= params["warmup_epocas"]:
            lr_atual = params["lr"] * epoca / params["warmup_epocas"]
            for grupo in otimizador.param_groups:
                grupo["lr"] = lr_atual

        aluno.train()
        total_loss = 0.0
        n_batches = 0

        for imgs, _ in loader_treino:
            imgs = imgs.to(dispositivo)
            otimizador.zero_grad(set_to_none=True)

            # Forward aluno e professor
            z_aluno = aluno.projetar(imgs)
            with torch.no_grad():
                z_professor = professor.projetar(imgs)

            # Loss: cross-entropy com centering e sharpening
            perda = _dino_loss(
                z_aluno, z_professor.detach(),
                params["temperatura_aluno"], params["temperatura_professor"],
            )
            perda.backward()
            nn.utils.clip_grad_norm_(aluno.parameters(), 1.0)
            otimizador.step()

            # EMA update do professor
            _atualizar_ema(aluno, professor, ema_momentum)

            total_loss += perda.item()
            n_batches += 1

        log.info("Época %3d/%d | loss %.4f", epoca, params["epocas"], total_loss / n_batches)

    torch.save(aluno.backbone.state_dict(), caminho_saida)
    log.info("Backbone salvo em %s", caminho_saida)
    return caminho_saida


def _dino_loss(
    z_aluno: torch.Tensor,
    z_professor: torch.Tensor,
    temp_aluno: float,
    temp_professor: float,
) -> torch.Tensor:
    """Perda DINO: cross-entropy entre distribuições suavizadas."""
    p_prof = F.softmax(z_professor / temp_professor, dim=-1)
    log_p_aluno = F.log_softmax(z_aluno / temp_aluno, dim=-1)
    return -(p_prof * log_p_aluno).sum(dim=-1).mean()


def _atualizar_ema(aluno: nn.Module, professor: nn.Module, momentum: float) -> None:
    """Atualiza os pesos do professor via EMA."""
    for p_aluno, p_prof in zip(aluno.parameters(), professor.parameters()):
        p_prof.data = momentum * p_prof.data + (1 - momentum) * p_aluno.data


if __name__ == "__main__":
    caminho = pre_treinar()
    print(f"Backbone salvo em: {caminho}")
