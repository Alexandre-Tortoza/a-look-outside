"""Extração de amostras por classe para datasets de imagem."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np


def extrair_amostras_por_classe(
    imagens: np.ndarray,
    rotulos: np.ndarray,
    n_por_classe: int = 50,
    semente: int = 42,
) -> dict[int, np.ndarray]:
    """Amostra aleatoriamente até n_por_classe imagens de cada classe.

    Args:
        imagens: Array (N, H, W, C) ou (N, H, W) de imagens.
        rotulos: Array (N,) de rótulos inteiros.
        n_por_classe: Número máximo de imagens por classe. Usa todas as disponíveis
            se a classe tiver menos, emitindo aviso.
        semente: Semente para reprodutibilidade da amostragem.

    Returns:
        Dicionário {classe_id: array (k, H, W, C)} com as imagens amostradas.
    """
    rng = np.random.default_rng(semente)
    classes = sorted(np.unique(rotulos).tolist())
    amostras: dict[int, np.ndarray] = {}

    for c in classes:
        indices = np.where(rotulos == c)[0]
        disponivel = len(indices)
        if disponivel < n_por_classe:
            warnings.warn(
                f"Classe {c}: solicitadas {n_por_classe} amostras, "
                f"disponíveis apenas {disponivel}. Usando todas as disponíveis.",
                stacklevel=2,
            )
            escolhidos = indices
        else:
            escolhidos = rng.choice(indices, size=n_por_classe, replace=False)

        amostras[c] = imagens[escolhidos]

    return amostras


def salvar_imagens_por_classe(
    amostras_por_classe: dict[int, np.ndarray],
    diretorio_saida: Path,
    nomes_classes: Optional[dict[int, str]] = None,
) -> dict[int, list[Path]]:
    """Salva imagens individuais em subpastas por classe.

    Estrutura gerada::

        diretorio_saida/
            classe_00/
                img_00000.png
                img_00001.png
                ...
            classe_01/
                ...

    Args:
        amostras_por_classe: Saída de extrair_amostras_por_classe().
        diretorio_saida: Diretório raiz onde criar as subpastas.
        nomes_classes: Mapeamento opcional {classe_id: nome_legível} para nomear as pastas.

    Returns:
        Dicionário {classe_id: [lista de Paths salvos]}.
    """
    from PIL import Image

    diretorio_saida = Path(diretorio_saida)
    caminhos: dict[int, list[Path]] = {}

    for c, imgs in sorted(amostras_por_classe.items()):
        nome_pasta = nomes_classes[c] if nomes_classes and c in nomes_classes else f"classe_{c:02d}"
        pasta_classe = diretorio_saida / nome_pasta
        pasta_classe.mkdir(parents=True, exist_ok=True)

        caminhos[c] = []
        for i, img in enumerate(imgs):
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)

            if img.ndim == 2:
                pil_img = Image.fromarray(img, mode="L")
            elif img.shape[-1] == 1:
                pil_img = Image.fromarray(img[..., 0], mode="L")
            else:
                pil_img = Image.fromarray(img[..., :3])

            caminho = pasta_classe / f"img_{i:05d}.png"
            pil_img.save(caminho)
            caminhos[c].append(caminho)

    return caminhos
