"""Fine-tuning de etapa unica do VGG16 a partir de um checkpoint existente.

Difere do treino padrao (treino.py):
- Carrega pesos de um .pth existente (nao ImageNet do zero)
- Nenhuma fase de congelamento: todos os layers treinados desde a epoca 1
- LRs muito baixos (backbone 5e-6, cabeca 1e-5 por padrao)
- Projetado para adaptar um modelo ja treinado a um novo dataset

Uso direto:
    python -m modelos.vgg16.finetuning --checkpoint pesos/vgg16/NOME.pth
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset.carregador import CarregadorDataset, resolver_nome_dataset
from dataset.dataset_pytorch import criar_dataloaders
from modelos._transfer_learning import _epoch_treino, _epoch_val
from modelos.treinador import HistoricoTreino
from modelos.vgg16.modelo import VGG16Galaxy
from pre_processamento.divisao_treino_teste import dividir_estratificado
from pre_processamento.normalizacao import obter_transform_avaliacao, obter_transform_treino
from utils.checkpoint import carregar_checkpoint, salvar_checkpoint
from utils.config_loader import carregar_config, obter_config_modelo, obter_config_recursos
from utils.experimento import registrar_run
from utils.logger import obter_logger
from utils.recursos import aplicar_batch_cap, configurar_dispositivo, obter_num_workers, usar_mixed_precision
from utils.reproducibilidade import fixar_semente
from utils.visualizacao import plotar_curva_treino


def fine_tuning(
    caminho_checkpoint: Path,
    config_override: Optional[dict[str, Any]] = None,
) -> HistoricoTreino:
    """Fine-tuning de etapa unica a partir de um checkpoint VGG16.

    Todos os parametros sao treinados desde a epoca 1 com LR diferenciado
    (backbone muito menor que a cabeca). O checkpoint de origem substitui
    os pesos ImageNet — ideal para adaptar um modelo ja especializado.

    Args:
        caminho_checkpoint: Caminho do .pth a usar como ponto de partida.
        config_override: Overrides de hiperparametros (dataset, epocas, lr, etc.).

    Returns:
        HistoricoTreino com metricas por epoca.
    """
    config = carregar_config()
    params = obter_config_modelo("vgg16", config)
    cfg_rec = obter_config_recursos(config)

    # Defaults especificos de fine-tuning (sobrescritos por config_override)
    params.setdefault("lr_backbone", 5e-6)
    params.setdefault("lr_cabeca", 1e-5)
    params.setdefault("epocas", 30)
    params.setdefault("label_smoothing", 0.05)
    params.setdefault("paciencia_early_stop", 7)
    params.setdefault("dataset", "fusao")
    params.setdefault("versao_dataset", "raw")

    if config_override:
        params.update(config_override)

    dataset = params["dataset"]
    versao = params["versao_dataset"]

    num_workers = obter_num_workers(cfg_rec)
    batch_size = aplicar_batch_cap(params.get("batch_size", 16), cfg_rec)
    params["batch_size"] = batch_size

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_exp = params.get("nome_experimento") or f"vgg16_ft_{dataset}_{versao}_ep{params['epocas']}_{timestamp}"
    params["nome_experimento"] = nome_exp
    params["checkpoint_origem"] = str(caminho_checkpoint)

    log = obter_logger(__name__, arquivo_log=Path("docs/vgg16/finetuning.log"))
    fixar_semente(params["seed"])

    # --- Dataset ---
    nome_dataset = resolver_nome_dataset(dataset, versao)
    imagens, rotulos = CarregadorDataset().carregar(nome_dataset)
    divisao = dividir_estratificado(imagens, rotulos, semente=params["seed"])

    loader_treino, loader_val, _ = criar_dataloaders(
        divisao,
        obter_transform_treino(tamanho_imagem=params["tamanho_imagem"]),
        obter_transform_avaliacao(tamanho_imagem=params["tamanho_imagem"]),
        batch_size, num_workers,
    )

    # --- Modelo: carrega checkpoint, nao usa ImageNet ---
    num_classes = len(set(rotulos.tolist()))
    rede = VGG16Galaxy(pretrained=False).construir(
        num_classes=num_classes, tamanho_imagem=params["tamanho_imagem"]
    )
    log.info("Carregando checkpoint de: %s", caminho_checkpoint)
    carregar_checkpoint(caminho_checkpoint, rede)

    dispositivo = configurar_dispositivo(cfg_rec)
    amp = usar_mixed_precision(cfg_rec, dispositivo)
    rede = rede.to(dispositivo)

    # --- Descongelar tudo com LR diferenciado ---
    for param in rede.parameters():
        param.requires_grad = True

    def _eh_cabeca(nome: str) -> bool:
        return any(parte in nome for parte in ("head", "fc", "classifier"))

    params_backbone = [p for n, p in rede.named_parameters() if not _eh_cabeca(n)]
    params_cabeca = [p for n, p in rede.named_parameters() if _eh_cabeca(n)]

    otimizador = AdamW([
        {"params": params_backbone, "lr": params["lr_backbone"]},
        {"params": params_cabeca,   "lr": params["lr_cabeca"]},
    ], weight_decay=params.get("peso_decay", 1e-4))

    epocas = params["epocas"]
    scheduler = CosineAnnealingLR(otimizador, T_max=epocas, eta_min=1e-7)
    scaler = torch.cuda.amp.GradScaler() if amp else None
    criterio = nn.CrossEntropyLoss(label_smoothing=params.get("label_smoothing", 0.0))

    # --- Loop de treino ---
    historico = HistoricoTreino()
    sem_melhora = 0
    paciencia = params.get("paciencia_early_stop", 7)

    pasta_pesos = Path("pesos") / "vgg16"
    pasta_pesos.mkdir(parents=True, exist_ok=True)
    caminho_pesos = pasta_pesos / f"{nome_exp}.pth"

    log.info(
        "Fine-tuning VGG16 | origem=%s | dataset=%s | "
        "lr_backbone=%.1e | lr_cabeca=%.1e | epocas=%d | batch=%d",
        Path(caminho_checkpoint).name, dataset,
        params["lr_backbone"], params["lr_cabeca"], epocas, batch_size,
    )

    for epoca in range(1, epocas + 1):
        loss_t, acc_t = _epoch_treino(rede, loader_treino, otimizador, criterio, scaler, dispositivo)
        loss_v, acc_v = _epoch_val(rede, loader_val, criterio, dispositivo)
        scheduler.step()

        historico.perdas_treino.append(loss_t)
        historico.perdas_val.append(loss_v)
        historico.accs_treino.append(acc_t)
        historico.accs_val.append(acc_v)
        historico.epocas_totais = epoca

        log.info("Época %3d/%d | loss %.4f | acc %.4f | val_loss %.4f | val_acc %.4f",
                 epoca, epocas, loss_t, acc_t, loss_v, acc_v)

        if acc_v > historico.melhor_val_acc:
            historico.melhor_val_acc = acc_v
            sem_melhora = 0
            if params.get("salvar_pesos", True):
                salvar_checkpoint(rede, caminho_pesos, params=params, historico=historico)
                log.info("  ✓ Melhor modelo salvo (val_acc=%.4f)", acc_v)
        else:
            sem_melhora += 1

        if sem_melhora >= paciencia:
            log.info("Early stopping após %d épocas sem melhora.", paciencia)
            historico.parou_cedo = True
            break

    # --- Curva e registro ---
    plotar_curva_treino(
        historico.perdas_treino, historico.perdas_val,
        historico.accs_treino, historico.accs_val,
        caminho_saida=Path("docs/vgg16/curva_finetuning.png"),
    )
    registrar_run(Path("pesos"), "vgg16", nome_exp, params, historico)

    log.info("Fine-tuning concluído. %s | checkpoint: %s", historico.resumo(), caminho_pesos)
    return historico


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning VGG16 a partir de checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Caminho do .pth a usar como ponto de partida")
    parser.add_argument("--dataset", default="fusao",
                        help="Dataset para fine-tuning (padrao: fusao)")
    parser.add_argument("--versao", default="raw", help="Versao do dataset (padrao: raw)")
    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--lr-backbone", type=float, default=5e-6)
    parser.add_argument("--lr-cabeca", type=float, default=1e-5)
    args = parser.parse_args()

    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    historico = fine_tuning(
        args.checkpoint,
        config_override={
            "dataset": args.dataset,
            "versao_dataset": args.versao,
            "epocas": args.epocas,
            "lr_backbone": args.lr_backbone,
            "lr_cabeca": args.lr_cabeca,
        },
    )
    print(historico.resumo())
