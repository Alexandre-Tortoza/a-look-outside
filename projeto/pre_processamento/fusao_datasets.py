"""Criacao do dataset fundido SDSS + DECaLS (somente imagens, sem metadados).

O arquivo gerado (fusao.h5) contem todas as imagens de ambos os datasets
redimensionadas para 224x224 LANCZOS, embaralhadas com semente fixa.
Metadados presentes no DECaLS (ra, dec, redshift, etc.) sao ignorados.

Uso direto:
    python -m pre_processamento.fusao_datasets
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

from dataset.carregador import CarregadorDataset
from pre_processamento.upscale import aplicar_upscale

_log = logging.getLogger(__name__)

CAMINHO_FUSAO = Path("dataset/processados/fusao.h5")
TAMANHO_FUSAO = 224


def criar_dataset_fusao(
    tamanho: int = TAMANHO_FUSAO,
    semente: int = 42,
    caminho_saida: Path = CAMINHO_FUSAO,
    forcar: bool = False,
) -> Path:
    """Combina SDSS e DECaLS (somente imagens/rotulos) num unico H5 em 224x224.

    SDSS (69px) e DECaLS (256px) sao redimensionados para ``tamanho`` via LANCZOS.
    Metadados do DECaLS sao ignorados — apenas as chaves ``images`` e ``ans`` sao lidas.

    Args:
        tamanho: Resolucao de saida (altura = largura). Padrao: 224.
        semente: Semente para shuffle reproducivel.
        caminho_saida: Onde salvar o H5 gerado.
        forcar: Se True, recria mesmo que o arquivo ja exista.

    Returns:
        Caminho do arquivo H5 gerado.
    """
    caminho_saida = Path(caminho_saida)

    if caminho_saida.exists() and not forcar:
        tam_mb = caminho_saida.stat().st_size / 1e6
        _log.info("Dataset fusao ja existe: %s (%.1f MB) — use forcar=True para recriar.",
                  caminho_saida, tam_mb)
        return caminho_saida

    carregador = CarregadorDataset()

    _log.info("Carregando SDSS...")
    imgs_sdss, rots_sdss = carregador.carregar("sdss")
    _log.info("SDSS: %d amostras, shape %s", len(rots_sdss), imgs_sdss.shape)

    _log.info("Carregando DECaLS (somente images/ans)...")
    imgs_decals, rots_decals = carregador.carregar("decals")
    _log.info("DECaLS: %d amostras, shape %s", len(rots_decals), imgs_decals.shape)

    # Garantir 3 canais (H, W) -> (H, W, 3)
    if imgs_sdss.ndim == 3:
        imgs_sdss = np.stack([imgs_sdss, imgs_sdss, imgs_sdss], axis=-1)
    if imgs_decals.ndim == 3:
        imgs_decals = np.stack([imgs_decals, imgs_decals, imgs_decals], axis=-1)

    # Redimensionar ambos para tamanho comum
    if imgs_sdss.shape[1] != tamanho or imgs_sdss.shape[2] != tamanho:
        _log.info("Upscale SDSS %dpx -> %dpx (LANCZOS)...", imgs_sdss.shape[1], tamanho)
        imgs_sdss = aplicar_upscale(imgs_sdss, tamanho_alvo=tamanho)

    if imgs_decals.shape[1] != tamanho or imgs_decals.shape[2] != tamanho:
        _log.info("Resize DECaLS %dpx -> %dpx (LANCZOS)...", imgs_decals.shape[1], tamanho)
        imgs_decals = aplicar_upscale(imgs_decals, tamanho_alvo=tamanho)

    # Concatenar e embaralhar
    imgs = np.concatenate([imgs_sdss, imgs_decals], axis=0)
    rots = np.concatenate([rots_sdss, rots_decals], axis=0)

    rng = np.random.default_rng(semente)
    idx = rng.permutation(len(rots))
    imgs = imgs[idx]
    rots = rots[idx]

    n_total = len(rots)
    n_sdss = len(rots_sdss)
    n_decals = len(rots_decals)
    _log.info("Fusao: %d amostras totais (SDSS=%d, DECaLS=%d), shape %s",
              n_total, n_sdss, n_decals, imgs.shape)

    # Salvar
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(caminho_saida, "w") as f:
        f.create_dataset("images", data=imgs, compression="gzip", compression_opts=4)
        f.create_dataset("ans", data=rots.astype(np.int64))
        # Metadata
        f.attrs["n_sdss"] = n_sdss
        f.attrs["n_decals"] = n_decals
        f.attrs["tamanho"] = tamanho
        f.attrs["semente"] = semente

    tam_mb = caminho_saida.stat().st_size / 1e6
    _log.info("Fusao salva em: %s (%.1f MB)", caminho_saida, tam_mb)
    return caminho_saida


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    forcar = "--forcar" in sys.argv
    criar_dataset_fusao(forcar=forcar)
