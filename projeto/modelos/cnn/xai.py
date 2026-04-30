"""XAI para o CNN Baseline — wrapper do Grad-CAM compartilhado."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from utils.xai_gradcam import grad_cam as _grad_cam


def grad_cam(
    rede: nn.Module,
    tensor_entrada: torch.Tensor,
    classe_alvo: Optional[int] = None,
    nome_camada: str = "bloco5.3",
) -> np.ndarray:
    """Gera mapa Grad-CAM para o CNN Baseline.

    Args:
        rede: RedeCNNBaseline em modo eval.
        tensor_entrada: Tensor (1, C, H, W).
        classe_alvo: Classe a explicar; None → usa predita.
        nome_camada: Camada alvo (padrão: último conv do bloco5).

    Returns:
        Array float32 (H, W) normalizado [0, 1].
    """
    h, w = tensor_entrada.shape[2], tensor_entrada.shape[3]
    return _grad_cam(rede, tensor_entrada, nome_camada, classe_alvo, tamanho_saida=(h, w))
