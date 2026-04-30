"""Avaliação do modelo Multimodal."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dataset.carregador import CarregadorDataset, resolver_nome_dataset
from modelos.multimodal.modelo import RedeMultimodal
from modelos.multimodal.treino import DatasetMultimodal, _extrair_features_tabulares
from pre_processamento.divisao_treino_teste import dividir_estratificado
from pre_processamento.normalizacao import obter_transform_avaliacao
from utils.checkpoint import carregar_checkpoint
from utils.config_loader import carregar_config, obter_config_modelo, obter_config_recursos
from utils.logger import obter_logger
from utils.metricas import CLASSES_GALAXY10, ResultadoAvaliacao, calcular_metricas, formatar_para_markdown
from utils.recursos import configurar_dispositivo, obter_num_workers
from utils.reproducibilidade import fixar_semente
from utils.visualizacao import plotar_matriz_confusao


def avaliar(caminho_pesos: Path, config_override: Optional[dict[str, Any]] = None) -> ResultadoAvaliacao:
    config = carregar_config()
    params = obter_config_modelo("multimodal", config)
    cfg_rec = obter_config_recursos(config)

    if config_override:
        params.update(config_override)

    dataset = params.get("dataset", "decals")
    versao = params.get("versao_dataset", "raw")
    features_tab = params.get("features_tabulares", ["ra", "dec", "redshift", "mag_r", "mag_g", "mag_z"])

    dir_docs = Path("docs/multimodal")
    log = obter_logger(__name__, arquivo_log=dir_docs / "avaliacao.log")
    fixar_semente(params["seed"])

    carregador = CarregadorDataset()
    nome_dataset = resolver_nome_dataset(dataset, versao)
    imagens, rotulos = carregador.carregar(nome_dataset)
    n = len(rotulos)

    caminho_h5 = carregador._resolver_caminho(nome_dataset)
    features_raw = _extrair_features_tabulares(caminho_h5, features_tab, n)

    divisao = dividir_estratificado(imagens, rotulos, semente=params["seed"])

    indices = np.arange(n)
    fracao_treino = len(divisao.rotulos_treino) / n
    idx_tr, idx_rest = train_test_split(indices, train_size=fracao_treino,
                                        stratify=rotulos, random_state=params["seed"])
    fracao_val = len(divisao.rotulos_val) / len(idx_rest)
    _, idx_te = train_test_split(idx_rest, train_size=fracao_val,
                                 stratify=rotulos[idx_rest], random_state=params["seed"])

    scaler = StandardScaler()
    scaler.fit(features_raw[idx_tr])
    feat_teste = scaler.transform(features_raw[idx_te])

    num_workers = obter_num_workers(cfg_rec)
    transform_val = obter_transform_avaliacao(tamanho_imagem=params.get("tamanho_imagem", 224))
    ds_teste = DatasetMultimodal(divisao.imagens_teste, feat_teste, divisao.rotulos_teste, transform_val)
    loader_teste = DataLoader(ds_teste, batch_size=params.get("batch_size", 32),
                              num_workers=num_workers, shuffle=False)

    num_classes = len(set(rotulos.tolist()))
    num_feat = feat_teste.shape[1]
    rede = RedeMultimodal(num_classes=num_classes, num_features_tabulares=num_feat, pretrained=False)
    carregar_checkpoint(caminho_pesos, rede)

    dispositivo = configurar_dispositivo(cfg_rec)
    rede = rede.to(dispositivo).eval()

    todos_y, todos_pred, todos_logits = [], [], []
    with torch.no_grad():
        for imgs, tabs, rots in loader_teste:
            imgs, tabs = imgs.to(dispositivo), tabs.to(dispositivo)
            logits = rede(imgs, tabs)
            todos_y.extend(rots.numpy())
            todos_pred.extend(logits.argmax(1).cpu().numpy())
            todos_logits.append(logits.cpu().numpy())

    y_true = np.array(todos_y)
    y_pred = np.array(todos_pred)
    y_logits = np.concatenate(todos_logits)

    resultado = calcular_metricas(y_true, y_pred, y_logits,
                                  nome_modelo="Multimodal",
                                  nome_experimento=params.get("nome_experimento", "multimodal"))
    log.info("Acurácia: %.4f | F1: %.4f", resultado.acuracia, resultado.f1_macro)

    dir_docs.mkdir(parents=True, exist_ok=True)
    plotar_matriz_confusao(resultado.matriz_confusao, CLASSES_GALAXY10,
                           caminho_saida=dir_docs / "matriz_confusao.png")
    (dir_docs / "resultados.md").write_text(formatar_para_markdown(resultado), encoding="utf-8")

    return resultado
