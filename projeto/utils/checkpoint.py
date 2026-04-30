"""Checkpoints ricos: salva/carrega pesos + metadata (hiperparametros, metricas, timestamp)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn


def salvar_checkpoint(
    rede: nn.Module,
    caminho: Path,
    params: Optional[dict[str, Any]] = None,
    historico: Optional[Any] = None,
) -> None:
    """Salva checkpoint rico: state_dict + metadata.

    Gera dois arquivos:
    - ``caminho.pth``: dict torch carregavel com state_dict + metadata
    - ``caminho.json``: metadata legivel sem precisar de PyTorch

    Args:
        rede: Modelo treinado.
        caminho: Caminho do .pth (ex: ``pesos/cnn/cnn_decals_raw_ep50_20260430.pth``).
        params: Hiperparametros usados no treino.
        historico: HistoricoTreino com metricas.
    """
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()

    checkpoint: dict[str, Any] = {"state_dict": rede.state_dict()}

    metadata: dict[str, Any] = {"timestamp": timestamp}

    if params is not None:
        # Remove valores nao serializaveis para o JSON
        params_limpo = _limpar_para_json(params)
        checkpoint["hiperparametros"] = params_limpo
        metadata["hiperparametros"] = params_limpo

    if historico is not None:
        metricas = {
            "melhor_val_acc": getattr(historico, "melhor_val_acc", None),
            "epocas_totais": getattr(historico, "epocas_totais", None),
            "parou_cedo": getattr(historico, "parou_cedo", None),
        }
        checkpoint["metricas"] = metricas
        metadata["metricas"] = metricas

    checkpoint["timestamp"] = timestamp

    torch.save(checkpoint, caminho)

    # JSON irmao legivel
    caminho_json = caminho.with_suffix(".json")
    caminho_json.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def carregar_checkpoint(
    caminho: str | Path,
    rede: nn.Module,
) -> dict[str, Any]:
    """Carrega checkpoint de forma backward-compatible.

    Aceita tanto:
    - state_dict puro (formato antigo): aplica direto no modelo
    - dict com chave ``"state_dict"`` (formato novo): extrai e aplica

    Args:
        caminho: Caminho do .pth.
        rede: Modelo onde carregar os pesos.

    Returns:
        Dict com metadata do checkpoint (vazio se formato antigo).
    """
    dados = torch.load(caminho, map_location="cpu", weights_only=False)

    if isinstance(dados, dict) and "state_dict" in dados:
        rede.load_state_dict(dados["state_dict"])
        return {k: v for k, v in dados.items() if k != "state_dict"}

    # Formato antigo: state_dict puro
    rede.load_state_dict(dados)
    return {}


def _limpar_para_json(d: dict) -> dict:
    """Remove valores nao serializaveis de um dict para salvar em JSON."""
    limpo = {}
    for k, v in d.items():
        if isinstance(v, dict):
            limpo[k] = _limpar_para_json(v)
        elif isinstance(v, (str, int, float, bool, type(None))):
            limpo[k] = v
        elif isinstance(v, (list, tuple)):
            limpo[k] = list(v)
        else:
            limpo[k] = str(v)
    return limpo
