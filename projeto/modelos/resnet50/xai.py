"""XAI para ResNet50 — wrapper do Grad-CAM compartilhado."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch.nn as nn
import torch

from utils.xai_gradcam import grad_cam as _grad_cam


def grad_cam(
    rede: nn.Module,
    tensor_entrada: torch.Tensor,
    classe_alvo: Optional[int] = None,
    nome_camada: str = "layer4.2.conv3",
) -> np.ndarray:
    h, w = tensor_entrada.shape[2], tensor_entrada.shape[3]
    return _grad_cam(rede, tensor_entrada, nome_camada, classe_alvo, tamanho_saida=(h, w))
