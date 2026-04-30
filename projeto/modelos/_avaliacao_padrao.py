"""Avaliação padrão reutilizável para todos os modelos com transfer learning."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from dataset.carregador import CarregadorDataset
from dataset.dataset_pytorch import criar_dataloaders
from modelos.base import ClassificadorGalaxias
from pre_processamento.divisao_treino_teste import dividir_estratificado
from pre_processamento.normalizacao import obter_transform_avaliacao, obter_transform_treino
from utils.logger import obter_logger
from utils.metricas import CLASSES_GALAXY10, ResultadoAvaliacao, calcular_metricas, formatar_para_markdown
from utils.reproducibilidade import fixar_semente
from utils.visualizacao import plotar_matriz_confusao


def avaliar_modelo(
    classificador: ClassificadorGalaxias,
    caminho_pesos: Path,
    dataset: str,
    tamanho_imagem: int,
    batch_size: int,
    seed: int,
    num_workers: int,
    nome_experimento: str,
    num_classes: int = 10,
) -> ResultadoAvaliacao:
    """Avaliação genérica aplicável a qualquer ClassificadorGalaxias.

    Args:
        classificador: Instância de ClassificadorGalaxias.
        caminho_pesos: Arquivo .pth com pesos.
        dataset: Nome do dataset ("sdss" | "decals").
        tamanho_imagem: Tamanho de entrada.
        batch_size: Tamanho do batch para inferência.
        seed: Semente para reprodutibilidade.
        num_workers: Workers do DataLoader.
        nome_experimento: Usado para nomear arquivos de saída.
        num_classes: Número de classes.

    Returns:
        ResultadoAvaliacao completo.
    """
    nome_modelo = classificador.nome.lower()
    dir_docs = Path("docs") / nome_modelo
    log = obter_logger(__name__, arquivo_log=dir_docs / "avaliacao.log")

    fixar_semente(seed)

    carregador = CarregadorDataset()
    imagens, rotulos = carregador.carregar(dataset)
    divisao = dividir_estratificado(imagens, rotulos, semente=seed)

    transform_val = obter_transform_avaliacao(tamanho_imagem=tamanho_imagem)
    _, _, loader_teste = criar_dataloaders(
        divisao,
        obter_transform_treino(tamanho_imagem=tamanho_imagem, aumentar=False),
        transform_val,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    rede = classificador.construir(num_classes=num_classes, tamanho_imagem=tamanho_imagem)
    rede.load_state_dict(torch.load(caminho_pesos, map_location="cpu"))
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rede = rede.to(dispositivo).eval()
    log.info("Avaliando %s com pesos de %s", classificador, caminho_pesos)

    todos_y, todos_pred, todos_logits = [], [], []
    with torch.no_grad():
        for imgs, rots in loader_teste:
            imgs = imgs.to(dispositivo)
            logits = rede(imgs)
            todos_y.extend(rots.numpy())
            todos_pred.extend(logits.argmax(1).cpu().numpy())
            todos_logits.append(logits.cpu().numpy())

    y_true = np.array(todos_y)
    y_pred = np.array(todos_pred)
    y_logits = np.concatenate(todos_logits)

    resultado = calcular_metricas(
        y_true, y_pred, y_logits,
        nome_modelo=classificador.nome,
        nome_experimento=nome_experimento,
    )
    log.info("Acurácia: %.4f | F1: %.4f", resultado.acuracia, resultado.f1_macro)

    dir_docs.mkdir(parents=True, exist_ok=True)
    plotar_matriz_confusao(resultado.matriz_confusao, CLASSES_GALAXY10,
                           caminho_saida=dir_docs / "matriz_confusao.png")
    (dir_docs / "resultados.md").write_text(formatar_para_markdown(resultado), encoding="utf-8")
    log.info("Resultados salvos em %s", dir_docs)

    return resultado


def inferir_modelo(
    classificador: ClassificadorGalaxias,
    imagem: np.ndarray,
    caminho_pesos: Path,
    num_classes: int = 10,
    tamanho_imagem: int = 224,
) -> tuple[int, float, np.ndarray]:
    """Inferência de imagem única genérica.

    Returns:
        (classe_prevista, confiança, logits)
    """
    from pre_processamento.normalizacao import obter_transform_avaliacao

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rede = classificador.construir(num_classes=num_classes, tamanho_imagem=tamanho_imagem)
    rede.load_state_dict(torch.load(caminho_pesos, map_location="cpu"))
    rede = rede.to(dispositivo).eval()

    transform = obter_transform_avaliacao(tamanho_imagem=tamanho_imagem)
    tensor = transform(imagem).unsqueeze(0).to(dispositivo)

    with torch.no_grad():
        logits = rede(tensor)
        probs = torch.softmax(logits, dim=1)
        classe = int(probs.argmax(1).item())
        confianca = float(probs[0, classe].item())

    return classe, confianca, logits.squeeze(0).cpu().numpy()
