"""Avaliação do ResNet50."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from modelos._avaliacao_padrao import avaliar_modelo
from modelos.resnet50 import config as cfg
from modelos.resnet50.modelo import ResNet50Galaxy
from utils.metricas import ResultadoAvaliacao


def avaliar(caminho_pesos: Path, config_override: Optional[dict] = None) -> ResultadoAvaliacao:
    params = {
        "dataset": cfg.DATASET, "tamanho_imagem": cfg.TAMANHO_IMAGEM,
        "batch_size": cfg.BATCH_SIZE, "seed": cfg.SEED, "num_workers": cfg.NUM_WORKERS,
        "nome_experimento": cfg.NOME_EXPERIMENTO, "pretrained": cfg.PRETRAINED,
    }
    if config_override:
        params.update(config_override)
    return avaliar_modelo(
        ResNet50Galaxy(pretrained=params["pretrained"]),
        caminho_pesos, params["dataset"], params["tamanho_imagem"],
        params["batch_size"], params["seed"], params["num_workers"], params["nome_experimento"],
    )


if __name__ == "__main__":
    import sys
    pesos = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(f"pesos/resnet50/{cfg.NOME_EXPERIMENTO}.pth")
    from utils.metricas import formatar_para_markdown
    print(formatar_para_markdown(avaliar(pesos)))
