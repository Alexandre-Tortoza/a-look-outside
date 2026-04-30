"""Utilitários compartilhados para modelos com transfer learning (two-stage fine-tuning)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from utils.logger import obter_logger
from utils.visualizacao import plotar_curva_treino
from modelos.treinador import HistoricoTreino


def treinar_two_stage(
    rede: nn.Module,
    nome_experimento: str,
    loader_treino: DataLoader,
    loader_val: DataLoader,
    epocas_total: int,
    epocas_congelado: int,
    lr_cabeca: float,
    lr_backbone: float,
    paciencia: int,
    salvar_checkpoints: bool,
    scheduler_ativo: bool,
    peso_decay: float,
    dispositivo: torch.device,
    dir_pesos: Path,
    dir_docs: Path,
    logger: Optional[logging.Logger] = None,
) -> HistoricoTreino:
    """Two-stage fine-tuning: primeiro treina só a cabeça, depois descongela tudo.

    Stage 1 (épocas 1..epocas_congelado): backbone congelado, só cabeça.
    Stage 2 (épocas seguintes): backbone descongelado com LR menor.

    Args:
        rede: Modelo com backbone + cabeça. Assume que rede.backbone (ou rede diretamente)
              e rede.head são acessíveis para congelar/descongelar.
              Usa heurística: congela todos params exceto os da última camada linear.
    """
    log = logger or obter_logger(__name__)
    criterio = nn.CrossEntropyLoss()
    usar_amp = dispositivo.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if usar_amp else None

    historico = HistoricoTreino()
    sem_melhora = 0
    nome_modelo = nome_experimento.split("_")[0]
    pasta_pesos = dir_pesos / nome_modelo
    pasta_pesos.mkdir(parents=True, exist_ok=True)
    caminho_pesos = pasta_pesos / f"{nome_experimento}.pth"

    rede = rede.to(dispositivo)

    for epoca in range(1, epocas_total + 1):
        # Troca de stage
        if epoca == 1:
            _congelar_backbone(rede)
            otimizador = AdamW(
                filter(lambda p: p.requires_grad, rede.parameters()),
                lr=lr_cabeca, weight_decay=peso_decay,
            )
            log.info("Stage 1: backbone congelado, treinando cabeça (LR=%.1e)", lr_cabeca)

        elif epoca == epocas_congelado + 1:
            _descongelar_tudo(rede)
            otimizador = AdamW([
                {"params": _params_backbone(rede), "lr": lr_backbone},
                {"params": _params_cabeca(rede), "lr": lr_cabeca},
            ], weight_decay=peso_decay)
            log.info("Stage 2: backbone descongelado (backbone LR=%.1e, cabeça LR=%.1e)",
                     lr_backbone, lr_cabeca)

        if scheduler_ativo and epoca == epocas_congelado + 1:
            scheduler = CosineAnnealingLR(otimizador, T_max=epocas_total - epocas_congelado, eta_min=1e-6)
        elif scheduler_ativo and epoca > epocas_congelado + 1:
            scheduler.step()

        loss_t, acc_t = _epoch_treino(rede, loader_treino, otimizador, criterio, scaler, dispositivo)
        loss_v, acc_v = _epoch_val(rede, loader_val, criterio, dispositivo)

        historico.perdas_treino.append(loss_t)
        historico.perdas_val.append(loss_v)
        historico.accs_treino.append(acc_t)
        historico.accs_val.append(acc_v)
        historico.epocas_totais = epoca

        log.info("Época %3d/%d | loss %.4f | acc %.4f | val_loss %.4f | val_acc %.4f",
                 epoca, epocas_total, loss_t, acc_t, loss_v, acc_v)

        if acc_v > historico.melhor_val_acc:
            historico.melhor_val_acc = acc_v
            sem_melhora = 0
            if salvar_checkpoints:
                torch.save(rede.state_dict(), caminho_pesos)
                log.info("  ✓ Melhor modelo salvo (val_acc=%.4f)", acc_v)
        else:
            sem_melhora += 1

        if sem_melhora >= paciencia:
            log.info("Early stopping após %d épocas sem melhora.", paciencia)
            historico.parou_cedo = True
            break

    # Curvas de treino
    dir_modelo = dir_docs / nome_modelo
    plotar_curva_treino(
        historico.perdas_treino, historico.perdas_val,
        historico.accs_treino, historico.accs_val,
        caminho_saida=dir_modelo / "curva_treino.png",
    )
    log.info("Treino concluído. %s", historico.resumo())
    return historico


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _congelar_backbone(rede: nn.Module) -> None:
    """Congela todos os parâmetros exceto a última camada linear."""
    for nome, param in rede.named_parameters():
        param.requires_grad = _eh_cabeca(nome, rede)


def _descongelar_tudo(rede: nn.Module) -> None:
    for param in rede.parameters():
        param.requires_grad = True


def _eh_cabeca(nome: str, rede: nn.Module) -> bool:
    """Heurística: considera cabeça os módulos chamados 'head', 'fc', 'classifier'."""
    return any(parte in nome for parte in ("head", "fc", "classifier"))


def _params_backbone(rede: nn.Module):
    return [p for n, p in rede.named_parameters() if not _eh_cabeca(n, rede)]


def _params_cabeca(rede: nn.Module):
    return [p for n, p in rede.named_parameters() if _eh_cabeca(n, rede)]


def _epoch_treino(rede, loader, otimizador, criterio, scaler, dispositivo):
    rede.train()
    total_loss, acertos, total = 0.0, 0, 0
    for imgs, rots in loader:
        imgs, rots = imgs.to(dispositivo, non_blocking=True), rots.to(dispositivo, non_blocking=True)
        otimizador.zero_grad(set_to_none=True)
        if scaler:
            with torch.cuda.amp.autocast():
                logits = rede(imgs)
                loss = criterio(logits, rots)
            scaler.scale(loss).backward()
            scaler.unscale_(otimizador)
            nn.utils.clip_grad_norm_(rede.parameters(), 1.0)
            scaler.step(otimizador)
            scaler.update()
        else:
            logits = rede(imgs)
            loss = criterio(logits, rots)
            loss.backward()
            nn.utils.clip_grad_norm_(rede.parameters(), 1.0)
            otimizador.step()
        bs = rots.size(0)
        total_loss += loss.item() * bs
        acertos += (logits.argmax(1) == rots).sum().item()
        total += bs
    return total_loss / total, acertos / total


def _epoch_val(rede, loader, criterio, dispositivo):
    rede.eval()
    total_loss, acertos, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, rots in loader:
            imgs, rots = imgs.to(dispositivo, non_blocking=True), rots.to(dispositivo, non_blocking=True)
            logits = rede(imgs)
            loss = criterio(logits, rots)
            bs = rots.size(0)
            total_loss += loss.item() * bs
            acertos += (logits.argmax(1) == rots).sum().item()
            total += bs
    return total_loss / total, acertos / total
