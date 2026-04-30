"""Aumento de dados offline: gera cópias aumentadas e salva em H5."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from PIL import Image, ImageEnhance

from pre_processamento.config import (
    BRILHO_CONTRASTE_FATOR,
    GRAU_ROTACAO_MAX,
    PROB_FLIP_HORIZONTAL,
    PROB_FLIP_VERTICAL,
    PROB_ROTACAO,
)


def _aumentar_imagem(imagem: np.ndarray, rng: random.Random) -> np.ndarray:
    """Aplica augmentation aleatório a uma única imagem (H, W, 3) uint8."""
    img = Image.fromarray(imagem)

    if rng.random() < PROB_FLIP_HORIZONTAL:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < PROB_FLIP_VERTICAL:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if rng.random() < PROB_ROTACAO:
        grau = rng.uniform(-GRAU_ROTACAO_MAX, GRAU_ROTACAO_MAX)
        img = img.rotate(grau, resample=Image.BILINEAR, fillcolor=0)

    fator_brilho = 1.0 + rng.uniform(-BRILHO_CONTRASTE_FATOR, BRILHO_CONTRASTE_FATOR)
    img = ImageEnhance.Brightness(img).enhance(fator_brilho)
    fator_contraste = 1.0 + rng.uniform(-BRILHO_CONTRASTE_FATOR, BRILHO_CONTRASTE_FATOR)
    img = ImageEnhance.Contrast(img).enhance(fator_contraste)

    return np.array(img)


def aplicar_aumento(
    imagens: np.ndarray,
    rotulos: np.ndarray,
    fator_multiplicacao: int = 2,
    semente: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Multiplica o dataset aplicando augmentation aleatório.

    Args:
        imagens: Array (N, H, W, 3) uint8.
        rotulos: Array (N,) int.
        fator_multiplicacao: Quantas cópias aumentadas gerar por imagem original.
        semente: Semente para reprodutibilidade.

    Returns:
        (imagens_aumentadas, rotulos_aumentados) — concatenação de original + cópias.
    """
    rng = random.Random(semente)
    imgs_extras = []
    rots_extras = []

    for _ in range(fator_multiplicacao - 1):
        for img, rot in zip(imagens, rotulos):
            imgs_extras.append(_aumentar_imagem(img, rng))
            rots_extras.append(rot)

    if imgs_extras:
        todas_imgs = np.concatenate([imagens, np.stack(imgs_extras)], axis=0)
        todos_rots = np.concatenate([rotulos, np.array(rots_extras)])
    else:
        todas_imgs = imagens
        todos_rots = rotulos

    return todas_imgs, todos_rots


def salvar_dataset_h5(
    imagens: np.ndarray,
    rotulos: np.ndarray,
    caminho: Path,
    nome_dataset: str = "processado",
) -> None:
    """Salva dataset em formato H5 compatível com CarregadorDataset.

    Chaves salvas: 'images' e 'ans' — mesma convenção dos arquivos Galaxy10 originais.

    Args:
        caminho: Caminho completo do arquivo .h5 de saída.
        nome_dataset: Metadado de identificação (salvo como atributo).
    """
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(caminho, "w") as f:
        f.create_dataset("images", data=imagens, compression="gzip", compression_opts=4)
        f.create_dataset("ans", data=rotulos.astype(np.int64))
        f.attrs["nome_dataset"] = nome_dataset
        f.attrs["n_amostras"] = len(rotulos)
