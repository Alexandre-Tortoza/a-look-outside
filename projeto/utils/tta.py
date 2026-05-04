"""Test-Time Augmentation (TTA) para ganho de acurácia sem retreino."""

from __future__ import annotations

import torch
import torch.nn as nn


def inferencia_tta(
    rede: nn.Module,
    tensor_base: torch.Tensor,
    n_augmentacoes: int = 5,
    dispositivo: torch.device | None = None,
) -> torch.Tensor:
    """Média de predições sob múltiplas augmentações geométricas.

    Augmentações aplicadas: original, flip-H, flip-V, rot90, rot180, rot270.
    Usa as primeiras ``n_augmentacoes + 1`` (inclui original).

    Args:
        rede: Modelo em modo eval (deve estar em eval() antes de chamar).
        tensor_base: Tensor (1, C, H, W) já pré-processado (normalizado).
        n_augmentacoes: Número de versões augmentadas além do original (máx 5).
        dispositivo: Device alvo. Se None, herda do primeiro parâmetro da rede.

    Returns:
        Tensor (num_classes,) com probabilidades médias.
    """
    if dispositivo is None:
        dispositivo = next(rede.parameters()).device

    augmentacoes = [
        lambda t: t,
        lambda t: t.flip(-1),
        lambda t: t.flip(-2),
        lambda t: torch.rot90(t, 1, [-2, -1]),
        lambda t: torch.rot90(t, 2, [-2, -1]),
        lambda t: torch.rot90(t, 3, [-2, -1]),
    ]
    usadas = augmentacoes[: min(n_augmentacoes + 1, len(augmentacoes))]

    probs_lista = []
    with torch.no_grad():
        for aug in usadas:
            x = aug(tensor_base.to(dispositivo))
            logits = rede(x)
            probs_lista.append(torch.softmax(logits, dim=-1))

    return torch.stack(probs_lista).mean(0).squeeze(0)  # (num_classes,)
