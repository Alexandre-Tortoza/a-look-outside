"""Knowledge Distillation para VGG16 usando ResNet50 como professor."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset.carregador import CarregadorDataset, resolver_nome_dataset
from dataset.dataset_pytorch import criar_dataloaders
from modelos.resnet50.modelo import ResNet50Galaxy
from modelos.vgg16.modelo import VGG16Galaxy
from modelos.treinador import HistoricoTreino
from pre_processamento.divisao_treino_teste import dividir_estratificado
from pre_processamento.normalizacao import obter_transform_avaliacao, obter_transform_treino
from utils.checkpoint import carregar_checkpoint, salvar_checkpoint
from utils.config_loader import carregar_config, obter_config_modelo, obter_config_recursos
from utils.experimento import gerar_nome_experimento, registrar_run
from utils.logger import obter_logger
from utils.recursos import aplicar_batch_cap, configurar_dispositivo, obter_num_workers, usar_mixed_precision
from utils.reproducibilidade import fixar_semente
from utils.visualizacao import plotar_curva_treino


def treinar_com_distilacao(
    config_override: Optional[dict[str, Any]] = None,
) -> HistoricoTreino:
    """Treina VGG16 com knowledge distillation usando ResNet50 como professor.

    O professor é congelado. A loss combina:
    - Hard loss: CrossEntropy(aluno, labels)
    - Soft loss: KL(softmax(aluno/T), softmax(professor/T))

    Hiperparâmetros relevantes no config.yaml (seção vgg16):
        distilacao_temperatura: temperatura T (padrão 4.0)
        distilacao_alpha: peso da soft loss (padrão 0.7)
        label_smoothing: aplicado na hard loss (padrão 0.05)
        paciencia_early_stop: early stop agressivo (padrão 5)
    """
    config = carregar_config()
    params = obter_config_modelo("vgg16", config)
    cfg_rec = obter_config_recursos(config)

    if config_override:
        params.update(config_override)

    dataset = params.get("dataset", "decals")
    versao = params.get("versao_dataset", "raw")
    params.setdefault("versao_dataset", versao)

    temperatura = params.get("distilacao_temperatura", 4.0)
    alpha = params.get("distilacao_alpha", 0.7)

    num_workers = obter_num_workers(cfg_rec)
    batch_size = aplicar_batch_cap(params.get("batch_size", 16), cfg_rec)
    params["batch_size"] = batch_size

    nome_exp = params.get("nome_experimento") or gerar_nome_experimento(
        "vgg16", dataset, versao, params["epocas"]
    )
    params["nome_experimento"] = nome_exp

    log = obter_logger(__name__, arquivo_log=Path("docs/vgg16/distilacao.log"))
    fixar_semente(params["seed"])

    nome_dataset = resolver_nome_dataset(dataset, versao)
    imagens, rotulos = CarregadorDataset().carregar(nome_dataset)
    divisao = dividir_estratificado(imagens, rotulos, semente=params["seed"])
    num_classes = len(set(rotulos.tolist()))

    loader_treino, loader_val, _ = criar_dataloaders(
        divisao,
        obter_transform_treino(tamanho_imagem=params["tamanho_imagem"]),
        obter_transform_avaliacao(tamanho_imagem=params["tamanho_imagem"]),
        batch_size, num_workers,
    )

    aluno = VGG16Galaxy(pretrained=params.get("pretrained", True)).construir(
        num_classes=num_classes, tamanho_imagem=params["tamanho_imagem"]
    )
    professor = ResNet50Galaxy(pretrained=True).construir(
        num_classes=num_classes, tamanho_imagem=params["tamanho_imagem"]
    )

    caminho_professor = _encontrar_pesos_recentes("resnet50")
    if caminho_professor:
        carregar_checkpoint(caminho_professor, professor)
        log.info("Professor (ResNet50) carregado de %s", caminho_professor)
    else:
        log.warning("Pesos de ResNet50 não encontrados. Professor usa apenas ImageNet.")

    dispositivo = configurar_dispositivo(cfg_rec)
    amp = usar_mixed_precision(cfg_rec, dispositivo)

    historico = _loop_distilacao(
        aluno=aluno, professor=professor,
        loader_treino=loader_treino, loader_val=loader_val,
        nome_exp=nome_exp, params=params,
        temperatura=temperatura, alpha=alpha,
        dispositivo=dispositivo, usar_amp=amp, log=log,
    )

    registrar_run(Path("pesos"), "vgg16", nome_exp, params, historico)
    return historico


# ---------------------------------------------------------------------------
# Loop interno
# ---------------------------------------------------------------------------

def _loop_distilacao(
    aluno: nn.Module,
    professor: nn.Module,
    loader_treino: DataLoader,
    loader_val: DataLoader,
    nome_exp: str,
    params: dict,
    temperatura: float,
    alpha: float,
    dispositivo: torch.device,
    usar_amp: bool,
    log: logging.Logger,
) -> HistoricoTreino:
    aluno = aluno.to(dispositivo)
    professor = professor.to(dispositivo).eval()
    for p in professor.parameters():
        p.requires_grad = False

    criterio_hard = nn.CrossEntropyLoss(label_smoothing=params.get("label_smoothing", 0.0))
    otimizador = AdamW(
        aluno.parameters(),
        lr=params["lr_cabeca"],
        weight_decay=params.get("peso_decay", 1e-4),
    )
    scheduler = (
        CosineAnnealingLR(otimizador, T_max=params["epocas"], eta_min=1e-6)
        if params.get("scheduler_ativo", True)
        else None
    )
    scaler = torch.cuda.amp.GradScaler() if usar_amp else None

    historico = HistoricoTreino()
    sem_melhora = 0
    pasta = Path("pesos/vgg16")
    pasta.mkdir(parents=True, exist_ok=True)
    caminho_pesos = pasta / f"{nome_exp}.pth"
    paciencia = params.get("paciencia_early_stop", 5)

    for epoca in range(1, params["epocas"] + 1):
        loss_t, acc_t = _epoch_treino(
            aluno, professor, loader_treino, otimizador,
            criterio_hard, scaler, dispositivo, temperatura, alpha,
        )
        loss_v, acc_v = _epoch_val(aluno, loader_val, criterio_hard, dispositivo)

        if scheduler:
            scheduler.step()

        historico.perdas_treino.append(loss_t)
        historico.perdas_val.append(loss_v)
        historico.accs_treino.append(acc_t)
        historico.accs_val.append(acc_v)
        historico.epocas_totais = epoca

        log.info(
            "Época %3d/%d | loss %.4f | acc %.4f | val_loss %.4f | val_acc %.4f",
            epoca, params["epocas"], loss_t, acc_t, loss_v, acc_v,
        )

        if acc_v > historico.melhor_val_acc:
            historico.melhor_val_acc = acc_v
            sem_melhora = 0
            if params.get("salvar_pesos", True):
                salvar_checkpoint(aluno, caminho_pesos, params=params, historico=historico)
                log.info("  ✓ Melhor modelo salvo (val_acc=%.4f)", acc_v)
        else:
            sem_melhora += 1

        if sem_melhora >= paciencia:
            log.info("Early stopping após %d épocas sem melhora.", paciencia)
            historico.parou_cedo = True
            break

    plotar_curva_treino(
        historico.perdas_treino, historico.perdas_val,
        historico.accs_treino, historico.accs_val,
        caminho_saida=Path("docs/vgg16/curva_distilacao.png"),
    )
    return historico


def _loss_distilacao(
    logits_aluno: torch.Tensor,
    logits_prof: torch.Tensor,
    rotulos: torch.Tensor,
    criterio_hard: nn.CrossEntropyLoss,
    temperatura: float,
    alpha: float,
) -> torch.Tensor:
    """alpha * KL_soft + (1-alpha) * CrossEntropy_hard."""
    loss_hard = criterio_hard(logits_aluno, rotulos)
    soft_aluno = F.log_softmax(logits_aluno / temperatura, dim=-1)
    soft_prof = F.softmax(logits_prof / temperatura, dim=-1)
    loss_soft = F.kl_div(soft_aluno, soft_prof, reduction="batchmean") * (temperatura ** 2)
    return alpha * loss_soft + (1.0 - alpha) * loss_hard


def _epoch_treino(aluno, professor, loader, otimizador, criterio, scaler, dispositivo, T, alpha):
    aluno.train()
    total_loss, acertos, total = 0.0, 0, 0
    for imgs, rots in loader:
        imgs = imgs.to(dispositivo, non_blocking=True)
        rots = rots.to(dispositivo, non_blocking=True)
        otimizador.zero_grad(set_to_none=True)

        with torch.no_grad():
            logits_prof = professor(imgs)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits_aluno = aluno(imgs)
                loss = _loss_distilacao(logits_aluno, logits_prof, rots, criterio, T, alpha)
            scaler.scale(loss).backward()
            scaler.unscale_(otimizador)
            nn.utils.clip_grad_norm_(aluno.parameters(), 1.0)
            scaler.step(otimizador)
            scaler.update()
        else:
            logits_aluno = aluno(imgs)
            loss = _loss_distilacao(logits_aluno, logits_prof, rots, criterio, T, alpha)
            loss.backward()
            nn.utils.clip_grad_norm_(aluno.parameters(), 1.0)
            otimizador.step()

        bs = rots.size(0)
        total_loss += loss.item() * bs
        acertos += (logits_aluno.argmax(1) == rots).sum().item()
        total += bs
    return total_loss / total, acertos / total


def _epoch_val(rede, loader, criterio, dispositivo):
    rede.eval()
    total_loss, acertos, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, rots in loader:
            imgs, rots = imgs.to(dispositivo), rots.to(dispositivo)
            logits = rede(imgs)
            loss = criterio(logits, rots)
            bs = rots.size(0)
            total_loss += loss.item() * bs
            acertos += (logits.argmax(1) == rots).sum().item()
            total += bs
    return total_loss / total, acertos / total


def _encontrar_pesos_recentes(modelo: str) -> Optional[Path]:
    dir_pesos = Path("pesos") / modelo
    if not dir_pesos.exists():
        return None
    arquivos = sorted(dir_pesos.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return arquivos[0] if arquivos else None
