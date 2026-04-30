"""Pré-treino self-supervised do DINO (destilação professor-aluno).

Só relevante quando config modo = "pre_treino".
Para a maioria dos experimentos, prefira o modo "hub" (DINOv2 pré-treinado).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.carregador import CarregadorDataset, resolver_nome_dataset
from dataset.dataset_pytorch import criar_dataloaders
from modelos.dino.modelo import DinoGalaxy, _ModeloDinoScratch
from pre_processamento.divisao_treino_teste import dividir_estratificado
from pre_processamento.normalizacao import obter_transform_treino
from utils.checkpoint import salvar_checkpoint
from utils.config_loader import carregar_config, obter_config_modelo, obter_config_recursos
from utils.experimento import gerar_nome_experimento
from utils.logger import obter_logger
from utils.recursos import aplicar_batch_cap, configurar_dispositivo, obter_num_workers
from utils.reproducibilidade import fixar_semente


def pre_treinar(config_override: Optional[dict[str, Any]] = None) -> Path:
    """Executa o pré-treino DINO self-supervised.

    Retorna o caminho para os pesos salvos do professor/backbone.
    """
    config = carregar_config()
    params = obter_config_modelo("dino", config)
    cfg_rec = obter_config_recursos(config)

    if config_override:
        params.update(config_override)

    dataset = params.get("dataset", "decals")
    versao = params.get("versao_dataset", "raw")
    epocas = params.get("epocas_pre_treino", 100)

    num_workers = obter_num_workers(cfg_rec)
    batch_size = aplicar_batch_cap(params.get("batch_size", 32), cfg_rec)

    nome_exp = gerar_nome_experimento("dino", dataset, versao, epocas) + "_pretreino"

    log = obter_logger(__name__, arquivo_log=Path("docs/dino/pre_treino.log"))
    log.info("Iniciando pré-treino DINO | épocas=%d", epocas)
    fixar_semente(params["seed"])

    nome_dataset = resolver_nome_dataset(dataset, versao)
    imagens, rotulos = CarregadorDataset().carregar(nome_dataset)
    divisao = dividir_estratificado(imagens, rotulos, semente=params["seed"])

    loader_treino, _, _ = criar_dataloaders(
        divisao,
        obter_transform_treino(tamanho_imagem=params["tamanho_imagem"]),
        obter_transform_treino(tamanho_imagem=params["tamanho_imagem"], aumentar=False),
        batch_size, num_workers,
    )

    dispositivo = configurar_dispositivo(cfg_rec)
    num_classes = len(set(rotulos.tolist()))

    tamanho_projecao = params.get("tamanho_projecao", 65536)
    backbone_scratch = params.get("backbone_scratch", "vit_small_patch16_224")

    aluno: _ModeloDinoScratch = DinoGalaxy(
        modo="pre_treino", backbone_scratch=backbone_scratch,
        tamanho_projecao=tamanho_projecao
    ).construir(num_classes=num_classes, tamanho_imagem=params["tamanho_imagem"])
    aluno = aluno.to(dispositivo)

    professor = copy.deepcopy(aluno)
    for param in professor.parameters():
        param.requires_grad = False

    lr = params.get("lr_pre_treino", 5e-4)
    warmup_epocas = params.get("warmup_epocas", 10)
    temp_prof = params.get("temperatura_professor", 0.04)
    temp_aluno = params.get("temperatura_aluno", 0.1)

    otimizador = torch.optim.AdamW(aluno.parameters(), lr=lr, weight_decay=1e-4)
    ema_momentum = 0.996

    caminho_saida = Path("pesos/dino") / f"{nome_exp}.pth"
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    for epoca in range(1, epocas + 1):
        if epoca <= warmup_epocas:
            lr_atual = lr * epoca / warmup_epocas
            for grupo in otimizador.param_groups:
                grupo["lr"] = lr_atual

        aluno.train()
        total_loss = 0.0
        n_batches = 0

        for imgs, _ in loader_treino:
            imgs = imgs.to(dispositivo)
            otimizador.zero_grad(set_to_none=True)

            z_aluno = aluno.projetar(imgs)
            with torch.no_grad():
                z_professor = professor.projetar(imgs)

            perda = _dino_loss(z_aluno, z_professor.detach(), temp_aluno, temp_prof)
            perda.backward()
            nn.utils.clip_grad_norm_(aluno.parameters(), 1.0)
            otimizador.step()

            _atualizar_ema(aluno, professor, ema_momentum)

            total_loss += perda.item()
            n_batches += 1

        log.info("Época %3d/%d | loss %.4f", epoca, epocas, total_loss / n_batches)

    torch.save(aluno.backbone.state_dict(), caminho_saida)
    log.info("Backbone salvo em %s", caminho_saida)
    return caminho_saida


def _dino_loss(z_aluno, z_professor, temp_aluno, temp_professor):
    p_prof = F.softmax(z_professor / temp_professor, dim=-1)
    log_p_aluno = F.log_softmax(z_aluno / temp_aluno, dim=-1)
    return -(p_prof * log_p_aluno).sum(dim=-1).mean()


def _atualizar_ema(aluno, professor, momentum):
    for p_aluno, p_prof in zip(aluno.parameters(), professor.parameters()):
        p_prof.data = momentum * p_prof.data + (1 - momentum) * p_aluno.data


if __name__ == "__main__":
    caminho = pre_treinar()
    print(f"Backbone salvo em: {caminho}")
