"""Avaliação do CNN Baseline: métricas, matriz de confusão e resultados.md."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from dataset.carregador import CarregadorDataset, resolver_nome_dataset
from dataset.dataset_pytorch import criar_dataloaders
from modelos._avaliacao_padrao import preparar_loader_teste
from modelos.cnn.modelo import CNNBaseline, RedeCNNBaseline
from pre_processamento.divisao_treino_teste import dividir_estratificado
from pre_processamento.normalizacao import obter_transform_avaliacao, obter_transform_treino
from utils.checkpoint import carregar_checkpoint
from utils.config_loader import carregar_config, obter_config_modelo, obter_config_recursos
from utils.logger import obter_logger
from utils.metricas import CLASSES_GALAXY10, ResultadoAvaliacao, calcular_metricas, formatar_para_markdown
from utils.recursos import configurar_dispositivo, obter_num_workers
from utils.reproducibilidade import fixar_semente
from utils.visualizacao import plotar_matriz_confusao


def avaliar(
    caminho_pesos: Path,
    config_override: Optional[dict[str, Any]] = None,
) -> ResultadoAvaliacao:
    """Avalia o CNN Baseline usando o conjunto de teste."""
    config = carregar_config()
    params = obter_config_modelo("cnn", config)
    cfg_rec = obter_config_recursos(config)

    if config_override:
        params.update(config_override)

    dataset = params.get("dataset", "decals")
    versao = params.get("versao_dataset", "raw")

    dir_docs = Path("docs/cnn")
    log = obter_logger(__name__, arquivo_log=dir_docs / "avaliacao.log")
    fixar_semente(params["seed"])

    num_workers = obter_num_workers(cfg_rec)
    tamanho_imagem = params.get("tamanho_imagem", 224)
    transform_val = obter_transform_avaliacao(tamanho_imagem=tamanho_imagem)

    carregador = CarregadorDataset()
    loader_teste, rotulos_ref = preparar_loader_teste(params, carregador, transform_val, num_workers)

    num_classes = len(set(rotulos_ref.tolist()))
    rede = RedeCNNBaseline(num_classes=num_classes)
    carregar_checkpoint(caminho_pesos, rede)

    dispositivo = configurar_dispositivo(cfg_rec)
    rede = rede.to(dispositivo).eval()
    log.info("Pesos carregados de %s", caminho_pesos)

    todos_y, todos_pred, todos_logits = [], [], []
    with torch.no_grad():
        for imgs, rots in loader_teste:
            imgs = imgs.to(dispositivo)
            logits = rede(imgs)
            todos_y.extend(rots.numpy())
            todos_pred.extend(logits.argmax(dim=1).cpu().numpy())
            todos_logits.append(logits.cpu().numpy())

    y_true = np.array(todos_y)
    y_pred = np.array(todos_pred)
    y_logits = np.concatenate(todos_logits)

    resultado = calcular_metricas(
        y_true, y_pred, y_logits,
        nome_modelo="CNN",
        nome_experimento=params.get("nome_experimento", "cnn"),
    )
    log.info("Acurácia: %.4f | F1: %.4f", resultado.acuracia, resultado.f1_macro)

    dir_docs.mkdir(parents=True, exist_ok=True)
    plotar_matriz_confusao(resultado.matriz_confusao, CLASSES_GALAXY10,
                           caminho_saida=dir_docs / "matriz_confusao.png")
    (dir_docs / "resultados.md").write_text(formatar_para_markdown(resultado), encoding="utf-8")

    return resultado
