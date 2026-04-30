"""XAI para DINO: mapas de atenção dos patch tokens."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mapas_atencao_dino(
    rede: nn.Module,
    tensor_entrada: torch.Tensor,
    cabeca_idx: Optional[int] = None,
    tamanho_saida: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    """Extrai mapas de atenção do último bloco do backbone DINO.

    O token CLS atende a todos os patch tokens — a distribuição de atenção
    forma um mapa espacial da "importância" de cada região da imagem.

    Args:
        rede: Modelo DINO (hub ou scratch) em modo eval.
        tensor_entrada: Tensor (1, C, H, W).
        cabeca_idx: Índice da cabeça de atenção. None → média de todas.
        tamanho_saida: (H, W) para upsample. None → tamanho dos patches.

    Returns:
        Array float32 (H, W) normalizado [0, 1].
    """
    rede.eval()
    dispositivo = next(rede.parameters()).device
    entrada = tensor_entrada.to(dispositivo)

    atencao_capturada: list[torch.Tensor] = []

    def _hook(modulo, inp, out):
        # DINOv2: o bloco de atenção retorna tensores intermediários
        # Tentamos capturar via atributo interno
        if hasattr(modulo, "attn"):
            pass  # tratado abaixo via hook no softmax
        atencao_capturada.append(out)

    # Tenta acessar o backbone diretamente
    backbone = getattr(rede, "backbone", rede)
    blocos = getattr(backbone, "blocks", None)

    if blocos is None or len(blocos) == 0:
        h_out = tamanho_saida[0] if tamanho_saida else tensor_entrada.shape[2]
        w_out = tamanho_saida[1] if tamanho_saida else tensor_entrada.shape[3]
        return np.ones((h_out, w_out), dtype=np.float32) / (h_out * w_out)

    ultimo_bloco = blocos[-1]
    atencao_bruta: list[torch.Tensor] = []

    def _hook_qkv(modulo, inp, out):
        # DINOv2 expõe q, k, v via 'qkv' linear
        # out: (B, N, 3*dim) ou similar
        atencao_bruta.append(out.detach())

    hook_qkv = None
    if hasattr(ultimo_bloco, "attn") and hasattr(ultimo_bloco.attn, "qkv"):
        hook_qkv = ultimo_bloco.attn.qkv.register_forward_hook(_hook_qkv)

    try:
        with torch.no_grad():
            rede(entrada)
    finally:
        if hook_qkv:
            hook_qkv.remove()

    if not atencao_bruta:
        h_out = tamanho_saida[0] if tamanho_saida else tensor_entrada.shape[2]
        w_out = tamanho_saida[1] if tamanho_saida else tensor_entrada.shape[3]
        return np.ones((h_out, w_out), dtype=np.float32) / (h_out * w_out)

    # Reconstruir matriz de atenção a partir de qkv
    qkv = atencao_bruta[0]  # (1, N, 3 * dim)
    B, N, qkv_dim = qkv.shape
    dim = qkv_dim // 3
    num_heads = getattr(ultimo_bloco.attn, "num_heads", 12)
    head_dim = dim // num_heads

    qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)  # (B, heads, N, head_dim)

    escala = head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * escala
    attn = torch.softmax(attn, dim=-1)  # (1, heads, N, N)

    if cabeca_idx is not None:
        mapa_attn = attn[0, cabeca_idx, 0, 1:].cpu().numpy()
    else:
        mapa_attn = attn[0, :, 0, 1:].mean(0).cpu().numpy()  # média sobre heads

    n_patches = mapa_attn.shape[0]
    n_lado = int(n_patches ** 0.5)

    if n_lado * n_lado != n_patches:
        h_out = tamanho_saida[0] if tamanho_saida else tensor_entrada.shape[2]
        w_out = tamanho_saida[1] if tamanho_saida else tensor_entrada.shape[3]
        return np.ones((h_out, w_out), dtype=np.float32) / (h_out * w_out)

    mapa = mapa_attn.reshape(n_lado, n_lado)
    mapa = (mapa - mapa.min()) / (mapa.max() - mapa.min() + 1e-8)

    if tamanho_saida is not None:
        tensor_mapa = torch.from_numpy(mapa).unsqueeze(0).unsqueeze(0)
        tensor_mapa = F.interpolate(tensor_mapa, size=tamanho_saida, mode="bilinear", align_corners=False)
        mapa = tensor_mapa.squeeze().numpy()

    return mapa.astype(np.float32)
