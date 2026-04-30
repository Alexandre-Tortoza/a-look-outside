"""Inferência individual com o modelo Multimodal."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from modelos.multimodal import config as cfg
from modelos.multimodal.modelo import RedeMultimodal
from pre_processamento.normalizacao import obter_transform_avaliacao


def inferir(
    imagem: np.ndarray,
    features_tabulares: np.ndarray,
    caminho_pesos: Path,
    num_classes: int = 10,
    tamanho_imagem: int = cfg.TAMANHO_IMAGEM,
) -> tuple[int, float, np.ndarray]:
    """Classifica uma imagem com seus metadados tabulares.

    Args:
        imagem: Array (H, W, 3) uint8.
        features_tabulares: Array (F,) float32 — features já normalizadas.
        caminho_pesos: Arquivo .pth.

    Returns:
        (classe_prevista, confiança, logits).
    """
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_feat = len(features_tabulares)

    rede = RedeMultimodal(num_classes=num_classes, num_features_tabulares=num_feat, pretrained=False)
    rede.load_state_dict(torch.load(caminho_pesos, map_location="cpu"))
    rede = rede.to(dispositivo).eval()

    transform = obter_transform_avaliacao(tamanho_imagem=tamanho_imagem)
    tensor_img = transform(imagem).unsqueeze(0).to(dispositivo)
    tensor_tab = torch.from_numpy(features_tabulares.astype(np.float32)).unsqueeze(0).to(dispositivo)

    with torch.no_grad():
        logits = rede(tensor_img, tensor_tab)
        probs = torch.softmax(logits, dim=1)
        classe = int(probs.argmax(1).item())
        confianca = float(probs[0, classe].item())

    return classe, confianca, logits.squeeze(0).cpu().numpy()
