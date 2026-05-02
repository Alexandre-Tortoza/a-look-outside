"""Controle global de aleatoriedade para experimentos reproduzíveis."""

import os
import random

import numpy as np
import torch

SEMENTE_GLOBAL: int = 42


def fixar_semente(semente: int = SEMENTE_GLOBAL, determinista: bool = True) -> None:
    """Fixa todas as fontes de aleatoriedade para garantir reprodutibilidade.

    Args:
        semente: Valor inteiro da semente global.
        determinista: Se True, força operações CUDA determinísticas (mais lento).
    """
    os.environ["PYTHONHASHSEED"] = str(semente)
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)

    if determinista:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
