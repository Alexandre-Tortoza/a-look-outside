"""Pipeline de treino do ResNet50."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from dataset.carregador import CarregadorDataset, resolver_nome_dataset
from dataset.dataset_pytorch import criar_dataloaders
from modelos._transfer_learning import treinar_two_stage
from modelos.resnet50.modelo import ResNet50Galaxy
from modelos.treinador import HistoricoTreino
from pre_processamento.divisao_treino_teste import dividir_estratificado
from pre_processamento.normalizacao import obter_transform_avaliacao, obter_transform_treino
from utils.config_loader import carregar_config, obter_config_modelo, obter_config_recursos
from utils.experimento import gerar_nome_experimento
from utils.logger import obter_logger
from utils.recursos import aplicar_batch_cap, configurar_dispositivo, obter_num_workers, usar_mixed_precision
from utils.reproducibilidade import fixar_semente


def treinar(config_override: Optional[dict[str, Any]] = None) -> HistoricoTreino:
    config = carregar_config()
    params = obter_config_modelo("resnet50", config)
    cfg_rec = obter_config_recursos(config)

    if config_override:
        params.update(config_override)

    dataset = params.get("dataset", "decals")
    versao = params.get("versao_dataset", "raw")
    params.setdefault("versao_dataset", versao)

    num_workers = obter_num_workers(cfg_rec)
    batch_size = aplicar_batch_cap(params.get("batch_size", 32), cfg_rec)
    params["batch_size"] = batch_size

    nome_exp = params.get("nome_experimento") or gerar_nome_experimento(
        "resnet50", dataset, versao, params["epocas"]
    )
    params["nome_experimento"] = nome_exp

    log = obter_logger(__name__, arquivo_log=Path("docs/resnet50/treino.log"))
    fixar_semente(params["seed"])

    nome_dataset = resolver_nome_dataset(dataset, versao)
    imagens, rotulos = CarregadorDataset().carregar(nome_dataset)
    divisao = dividir_estratificado(imagens, rotulos, semente=params["seed"])

    loader_treino, loader_val, _ = criar_dataloaders(
        divisao,
        obter_transform_treino(tamanho_imagem=params["tamanho_imagem"]),
        obter_transform_avaliacao(tamanho_imagem=params["tamanho_imagem"]),
        batch_size, num_workers,
    )

    num_classes = len(set(rotulos.tolist()))
    rede = ResNet50Galaxy(pretrained=params.get("pretrained", True)).construir(
        num_classes=num_classes, tamanho_imagem=params["tamanho_imagem"]
    )

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
    print(treinar().resumo())
