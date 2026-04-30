"""Inferência individual com o CNN Baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from modelos.cnn import config as cfg
from modelos.cnn.modelo import RedeCNNBaseline
from pre_processamento.normalizacao import obter_transform_avaliacao


def inferir(
    imagem: np.ndarray,
    caminho_pesos: Path,
    num_classes: int = 10,
    tamanho_imagem: int = cfg.TAMANHO_IMAGEM,
) -> tuple[int, float, np.ndarray]:
    """Classifica uma única imagem.

    Args:
        imagem: Array (H, W, 3) uint8.
        caminho_pesos: Caminho do arquivo .pth.
        num_classes: Número de classes do modelo salvo.
        tamanho_imagem: Tamanho de entrada esperado pelo modelo.

    Returns:
        (classe_prevista, confiança, logits) — logits shape (num_classes,).
    """
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rede = RedeCNNBaseline(num_classes=num_classes)
    rede.load_state_dict(torch.load(caminho_pesos, map_location="cpu"))
    rede = rede.to(dispositivo).eval()

    transform = obter_transform_avaliacao(tamanho_imagem=tamanho_imagem)
    tensor = transform(imagem).unsqueeze(0).to(dispositivo)

    with torch.no_grad():
        logits = rede(tensor)
        probs = torch.softmax(logits, dim=1)
        classe = int(probs.argmax(dim=1).item())
        confianca = float(probs[0, classe].item())

    return classe, confianca, logits.squeeze(0).cpu().numpy()
