"""Avaliação padrão reutilizável para todos os modelos com transfer learning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.carregador import CarregadorDataset, resolver_nome_dataset
from dataset.dataset_pytorch import DatasetGalaxias, criar_dataloaders
from modelos.base import ClassificadorGalaxias
from pre_processamento.divisao_treino_teste import dividir_estratificado
from pre_processamento.normalizacao import obter_transform_avaliacao, obter_transform_treino
from utils.checkpoint import carregar_checkpoint
from utils.config_loader import carregar_config, obter_config_modelo, obter_config_recursos
from utils.logger import obter_logger
from utils.metricas import CLASSES_GALAXY10, ResultadoAvaliacao, calcular_metricas, formatar_para_markdown
from utils.recursos import configurar_dispositivo, obter_num_workers
from utils.reproducibilidade import fixar_semente
from utils.visualizacao import plotar_matriz_confusao


def preparar_loader_teste(
    params: dict[str, Any],
    carregador: CarregadorDataset,
    transform_val,
    num_workers: int,
) -> tuple[DataLoader, np.ndarray]:
    """Prepara o DataLoader de teste, com suporte a cross-dataset.

    Se ``dataset_teste`` estiver em params, carrega esse dataset inteiro como teste.
    Senao, faz o split estratificado normal e retorna o loader de teste.

    Returns:
        (loader_teste, rotulos_originais_completo) — rotulos para contar num_classes.
    """
    dataset_teste = params.get("dataset_teste")

    if dataset_teste:
        # Cross-dataset: carrega dataset inteiro como teste
        imgs_teste, rots_teste = carregador.carregar(dataset_teste)
        ds_teste = DatasetGalaxias(imgs_teste, rots_teste, transform_val)
        loader = DataLoader(ds_teste, batch_size=params.get("batch_size", 32),
                            num_workers=num_workers, shuffle=False)
        return loader, rots_teste

    # Normal: split do dataset de treino
    dataset = params.get("dataset", "decals")
    versao = params.get("versao_dataset", "raw")
    nome_dataset = resolver_nome_dataset(dataset, versao)
    imagens, rotulos = carregador.carregar(nome_dataset)
    divisao = dividir_estratificado(imagens, rotulos, semente=params.get("seed", 42))
    _, _, loader_teste = criar_dataloaders(
        divisao,
        obter_transform_treino(tamanho_imagem=params.get("tamanho_imagem", 224), aumentar=False),
        transform_val,
        batch_size=params.get("batch_size", 32),
        num_workers=num_workers,
    )
    return loader_teste, rotulos


def avaliar_modelo(
    classificador: ClassificadorGalaxias,
    caminho_pesos: Path,
    params: dict[str, Any],
) -> ResultadoAvaliacao:
    """Avaliação genérica aplicável a qualquer ClassificadorGalaxias.

    Args:
        classificador: Instância de ClassificadorGalaxias.
        caminho_pesos: Arquivo .pth com pesos.
        params: Dict com dataset, tamanho_imagem, batch_size, seed, num_workers, etc.
    """
    config = carregar_config()
    cfg_rec = obter_config_recursos(config)

    nome_modelo = classificador.nome.lower()
    dir_docs = Path("docs") / nome_modelo
    log = obter_logger(__name__, arquivo_log=dir_docs / "avaliacao.log")

    fixar_semente(params.get("seed", 42))

    num_workers = obter_num_workers(cfg_rec)
    tamanho_imagem = params.get("tamanho_imagem", 224)
    num_classes = params.get("num_classes", 10)

    transform_val = obter_transform_avaliacao(tamanho_imagem=tamanho_imagem)

    carregador = CarregadorDataset()
    loader_teste, rotulos_ref = preparar_loader_teste(params, carregador, transform_val, num_workers)

    num_classes = len(set(rotulos_ref.tolist()))
    rede = classificador.construir(num_classes=num_classes, tamanho_imagem=tamanho_imagem)

    # Carregar checkpoint (backward-compatible)
    carregar_checkpoint(caminho_pesos, rede)

    dispositivo = configurar_dispositivo(cfg_rec)
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

    nome_exp = params.get("nome_experimento", nome_modelo)
    resultado = calcular_metricas(
        y_true, y_pred, y_logits,
        nome_modelo=classificador.nome,
        nome_experimento=nome_exp,
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
    """Inferência de imagem única genérica."""
    config = carregar_config()
    cfg_rec = obter_config_recursos(config)
    dispositivo = configurar_dispositivo(cfg_rec)

    rede = classificador.construir(num_classes=num_classes, tamanho_imagem=tamanho_imagem)
    carregar_checkpoint(caminho_pesos, rede)
    rede = rede.to(dispositivo).eval()

    transform = obter_transform_avaliacao(tamanho_imagem=tamanho_imagem)
    tensor = transform(imagem).unsqueeze(0).to(dispositivo)

    with torch.no_grad():
        logits = rede(tensor)
        probs = torch.softmax(logits, dim=1)
        classe = int(probs.argmax(1).item())
        confianca = float(probs[0, classe].item())

    return classe, confianca, logits.squeeze(0).cpu().numpy()
