"""Implementação model-agnostic de Grad-CAM via hooks nativos do PyTorch."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _obter_modulo(rede: nn.Module, nome: str) -> nn.Module:
    """Navega a árvore de módulos por notação de pontos.

    Args:
        rede: Módulo raiz.
        nome: Caminho separado por pontos, ex: "layer4.2.conv3".

    Returns:
        O submódulo indicado.

    Raises:
        AttributeError: Se o caminho não existir.
    """
    modulo = rede
    for parte in nome.split("."):
        if parte.isdigit():
            modulo = modulo[int(parte)]  # type: ignore[index]
        else:
            modulo = getattr(modulo, parte)
    return modulo


def grad_cam(
    rede: nn.Module,
    tensor_entrada: torch.Tensor,
    nome_camada: str,
    classe_alvo: Optional[int] = None,
    tamanho_saida: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Gera mapa de saliência Grad-CAM para uma imagem.

    Args:
        rede: Modelo PyTorch (nn.Module).
        tensor_entrada: Tensor (1, C, H, W) normalizado.
        nome_camada: Caminho em notação de pontos para a camada alvo (última conv).
        classe_alvo: Índice da classe para explicar. Se None, usa a predita.
        tamanho_saida: (H, W) para redimensionar o mapa final. Se None, usa (H, W) da entrada.

    Returns:
        Array float32 (H, W) normalizado em [0, 1].
    """
    rede.eval()
    dispositivo = next(rede.parameters()).device
    entrada = tensor_entrada.to(dispositivo)

    ativacoes: list[torch.Tensor] = []
    gradientes: list[torch.Tensor] = []

    camada = _obter_modulo(rede, nome_camada)

    def salvar_ativacao(_, __, saida: torch.Tensor) -> None:
        ativacoes.append(saida.detach())

    def salvar_gradiente(_, grad_entrada, grad_saida) -> None:
        gradientes.append(grad_saida[0].detach())

    hook_fwd = camada.register_forward_hook(salvar_ativacao)
    hook_bwd = camada.register_full_backward_hook(salvar_gradiente)

    try:
        logits = rede(entrada)
        if classe_alvo is None:
            classe_alvo = int(logits.argmax(dim=1).item())

        rede.zero_grad()
        logits[0, classe_alvo].backward()
    finally:
        hook_fwd.remove()
        hook_bwd.remove()

    ativ = ativacoes[0]     # (1, C, H', W')
    grad = gradientes[0]    # (1, C, H', W')

    # Pesos: média global dos gradientes por canal
    pesos = grad.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

    # Combinação linear ponderada + ReLU
    mapa = (pesos * ativ).sum(dim=1, keepdim=True)  # (1, 1, H', W')
    mapa = F.relu(mapa)

    # Normalização para [0, 1]
    minv = mapa.min()
    maxv = mapa.max()
    if maxv > minv:
        mapa = (mapa - minv) / (maxv - minv)
    else:
        mapa = torch.zeros_like(mapa)

    # Redimensiona para o tamanho desejado
    if tamanho_saida is None:
        h, w = tensor_entrada.shape[2], tensor_entrada.shape[3]
        tamanho_saida = (h, w)

    mapa = F.interpolate(mapa, size=tamanho_saida, mode="bilinear", align_corners=False)
    return mapa.squeeze().cpu().numpy().astype(np.float32)
