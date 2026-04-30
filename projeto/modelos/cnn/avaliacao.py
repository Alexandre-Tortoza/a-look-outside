"""Avaliação do CNN Baseline: métricas, matriz de confusão e resultados.md."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.carregador import CarregadorDataset
from dataset.dataset_pytorch import DatasetGalaxias, criar_dataloaders
from modelos.cnn import config as cfg
from modelos.cnn.modelo import CNNBaseline, RedeCNNBaseline
from pre_processamento.divisao_treino_teste import dividir_estratificado
from pre_processamento.normalizacao import obter_transform_avaliacao, obter_transform_treino
from utils.logger import obter_logger
from utils.metricas import CLASSES_GALAXY10, ResultadoAvaliacao, calcular_metricas, formatar_para_markdown
from utils.reproducibilidade import fixar_semente
from utils.visualizacao import plotar_matriz_confusao


def avaliar(
    caminho_pesos: Path,
    config_override: Optional[dict] = None,
) -> ResultadoAvaliacao:
    """Avalia o CNN Baseline usando o conjunto de teste.

    Args:
        caminho_pesos: Caminho do arquivo .pth com os pesos salvos.
        config_override: Sobrescreve parâmetros do config.py.

    Returns:
        ResultadoAvaliacao com todas as métricas.
    """
    params = {
        "dataset": cfg.DATASET,
        "tamanho_imagem": cfg.TAMANHO_IMAGEM,
        "batch_size": cfg.BATCH_SIZE,
        "seed": cfg.SEED,
        "num_workers": cfg.NUM_WORKERS,
        "nome_experimento": cfg.NOME_EXPERIMENTO,
    }
    if config_override:
        params.update(config_override)

    nome_modelo = "cnn"
    dir_docs = Path("docs") / nome_modelo
    log = obter_logger(__name__, arquivo_log=dir_docs / "avaliacao.log")

    fixar_semente(params["seed"])

    # Dataset + split
    carregador = CarregadorDataset()
    imagens, rotulos = carregador.carregar(params["dataset"])
    divisao = dividir_estratificado(imagens, rotulos, semente=params["seed"])

    transform_val = obter_transform_avaliacao(tamanho_imagem=params["tamanho_imagem"])
    _, _, loader_teste = criar_dataloaders(
        divisao,
        obter_transform_treino(tamanho_imagem=params["tamanho_imagem"], aumentar=False),
        transform_val,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"],
    )

    # Carregar modelo
    num_classes = len(set(rotulos.tolist()))
    rede = RedeCNNBaseline(num_classes=num_classes)
    rede.load_state_dict(torch.load(caminho_pesos, map_location="cpu"))
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rede = rede.to(dispositivo).eval()
    log.info("Pesos carregados de %s", caminho_pesos)

    # Inferência
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
        nome_experimento=params["nome_experimento"],
    )
    log.info("Acurácia: %.4f | F1: %.4f", resultado.acuracia, resultado.f1_macro)

    # Salvar outputs
    dir_docs.mkdir(parents=True, exist_ok=True)
    plotar_matriz_confusao(
        resultado.matriz_confusao,
        CLASSES_GALAXY10,
        caminho_saida=dir_docs / "matriz_confusao.png",
    )
    (dir_docs / "resultados.md").write_text(formatar_para_markdown(resultado), encoding="utf-8")
    log.info("Resultados salvos em %s", dir_docs)

    return resultado


if __name__ == "__main__":
    import sys
    pesos = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pesos/cnn") / f"{cfg.NOME_EXPERIMENTO}.pth"
    resultado = avaliar(pesos)
    print(formatar_para_markdown(resultado))
