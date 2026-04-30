"""Fine-tuning supervisionado após pré-treino DINO."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from dataset.carregador import CarregadorDataset, resolver_nome_dataset
from dataset.dataset_pytorch import criar_dataloaders
from modelos._transfer_learning import treinar_two_stage
from modelos.dino.modelo import DinoGalaxy
from modelos.treinador import HistoricoTreino
from pre_processamento.divisao_treino_teste import dividir_estratificado
from pre_processamento.normalizacao import obter_transform_avaliacao, obter_transform_treino
from utils.config_loader import carregar_config, obter_config_modelo, obter_config_recursos
from utils.experimento import gerar_nome_experimento
from utils.logger import obter_logger
from utils.recursos import aplicar_batch_cap, configurar_dispositivo, obter_num_workers, usar_mixed_precision
from utils.reproducibilidade import fixar_semente


def ajustar_fino(
    caminho_pesos_pretreino: Optional[Path] = None,
    config_override: Optional[dict[str, Any]] = None,
) -> HistoricoTreino:
    """Fine-tuning supervisionado do backbone DINO."""
    config = carregar_config()
    params = obter_config_modelo("dino", config)
    cfg_rec = obter_config_recursos(config)

    if config_override:
        params.update(config_override)

    dataset = params.get("dataset", "decals")
    versao = params.get("versao_dataset", "raw")
    params.setdefault("versao_dataset", versao)

    # DINO usa epocas_ajuste_fino como default de epocas
    params.setdefault("epocas", params.get("epocas_ajuste_fino", 30))
    params.setdefault("lr_cabeca", params.get("lr_cabeca", 1e-4))
    params.setdefault("lr_backbone", params.get("lr_backbone", 1e-5))

    num_workers = obter_num_workers(cfg_rec)
    batch_size = aplicar_batch_cap(params.get("batch_size", 32), cfg_rec)
    params["batch_size"] = batch_size

    nome_exp = params.get("nome_experimento") or gerar_nome_experimento(
        "dino", dataset, versao, params["epocas"]
    )
    params["nome_experimento"] = nome_exp

    log = obter_logger(__name__, arquivo_log=Path("docs/dino/ajuste_fino.log"))
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

    classificador = DinoGalaxy(
        modo=params.get("modo", "hub"),
        backbone_hub=params.get("backbone", "dinov2_vitb14"),
        backbone_scratch=params.get("backbone_scratch", "vit_small_patch16_224"),
    )
    rede = classificador.construir(num_classes=num_classes, tamanho_imagem=params["tamanho_imagem"])

    if caminho_pesos_pretreino is not None and caminho_pesos_pretreino.exists():
        backbone = getattr(rede, "backbone", rede)
        backbone.load_state_dict(torch.load(caminho_pesos_pretreino, map_location="cpu"))
        log.info("Backbone carregado de %s", caminho_pesos_pretreino)

    dispositivo = configurar_dispositivo(cfg_rec)
    amp = usar_mixed_precision(cfg_rec, dispositivo)

    return treinar_two_stage(
        rede=rede, nome_experimento=nome_exp,
        loader_treino=loader_treino, loader_val=loader_val,
        epocas_total=params["epocas"],
        epocas_congelado=params.get("epocas_congelado", 5),
        lr_cabeca=params["lr_cabeca"], lr_backbone=params["lr_backbone"],
        paciencia=params.get("paciencia_early_stop", 10),
        salvar_checkpoints=params.get("salvar_pesos", True),
        scheduler_ativo=params.get("scheduler_ativo", True),
        peso_decay=params.get("peso_decay", 1e-4),
        dispositivo=dispositivo, dir_pesos=Path("pesos"), dir_docs=Path("docs"),
        usar_amp=amp, params=params, logger=log,
    )


if __name__ == "__main__":
    historico = ajustar_fino()
    print(historico.resumo())
