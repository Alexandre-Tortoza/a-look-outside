"""XAI para Multimodal: Grad-CAM (branch visual) + SHAP (branch tabular)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from utils.xai_gradcam import grad_cam as _grad_cam


def grad_cam_multimodal(
    rede: nn.Module,
    tensor_imagem: torch.Tensor,
    classe_alvo: Optional[int] = None,
    nome_camada: str = "branch_visual.backbone.blocks.6.0.conv_pwl",
    tamanho_saida: Optional[tuple[int, int]] = None,
    tabular_dummy: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """Grad-CAM sobre o branch visual da RedeMultimodal.

    Como a rede multimodal requer dois inputs, envolve o forward em um wrapper
    que fixa o input tabular e expõe apenas a imagem.

    Args:
        rede: RedeMultimodal em modo eval.
        tensor_imagem: Tensor (1, C, H, W).
        classe_alvo: Classe a explicar.
        nome_camada: Camada alvo no branch visual.
        tamanho_saida: Tamanho do mapa de saída.
        tabular_dummy: Features tabulares de referência. Se None, usa zeros.

    Returns:
        Array float32 (H, W) normalizado [0, 1].
    """
    dispositivo = next(rede.parameters()).device
    entrada = tensor_imagem.to(dispositivo)

    if tabular_dummy is None:
        num_feat = rede.branch_tabular.mlp[0].in_features
        tabular_dummy = torch.zeros(1, num_feat, device=dispositivo)
    else:
        tabular_dummy = tabular_dummy.to(dispositivo)

    # Wrapper que fixa o input tabular
    class _WrapperVisual(nn.Module):
        def __init__(self, rede_multi, tab):
            super().__init__()
            self.rede = rede_multi
            self.tab = tab

        def forward(self, img):
            return self.rede(img, self.tab)

    wrapper = _WrapperVisual(rede, tabular_dummy)

    h, w = tensor_imagem.shape[2], tensor_imagem.shape[3]
    saida = tamanho_saida or (h, w)
    return _grad_cam(wrapper, entrada, nome_camada, classe_alvo, saida)


def shap_tabular(
    rede: nn.Module,
    features_background: np.ndarray,
    amostra: np.ndarray,
    dispositivo: str = "cpu",
) -> np.ndarray:
    """Gera valores SHAP para o branch tabular da RedeMultimodal.

    Usa shap.DeepExplainer sobre o MLP tabular com input visual zerado.

    Args:
        rede: RedeMultimodal.
        features_background: Array (N_bg, F) float32 — conjunto de referência.
        amostra: Array (1, F) float32 — amostra a explicar.
        dispositivo: "cpu" recomendado para SHAP.

    Returns:
        Array (F,) com os valores SHAP por feature.
    """
    try:
        import shap  # type: ignore
    except ImportError:
        raise ImportError("shap não instalado. Execute: pip install shap")

    dev = torch.device(dispositivo)
    rede = rede.to(dev).eval()

    # MLP wrapper: só o branch tabular
    class _BranchTabularWrapper(nn.Module):
        def __init__(self, rede_multi):
            super().__init__()
            self.branch = rede_multi.branch_tabular

        def forward(self, x):
            return self.branch(x)

    wrapper = _BranchTabularWrapper(rede)

    bg_tensor = torch.from_numpy(features_background).float().to(dev)
    amostra_tensor = torch.from_numpy(amostra).float().to(dev)

    explainer = shap.DeepExplainer(wrapper, bg_tensor)
    shap_values = explainer.shap_values(amostra_tensor)

    if isinstance(shap_values, list):
        # Multiclasse: média dos valores absolutos sobre classes
        return np.mean([np.abs(sv) for sv in shap_values], axis=0).squeeze()
    return np.abs(shap_values).squeeze()
