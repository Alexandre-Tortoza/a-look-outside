"""Funções de visualização reutilizáveis para treino, avaliação e XAI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # backend não-interativo, seguro em servidores

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils.metricas import ResultadoAvaliacao


def plotar_curva_treino(
    historico_loss_treino: list[float],
    historico_loss_val: list[float],
    historico_acc_treino: list[float],
    historico_acc_val: list[float],
    caminho_saida: Path,
    dpi: int = 150,
) -> None:
    """Salva curvas de loss e acurácia de treino/validação.

    Args:
        caminho_saida: Caminho completo do arquivo PNG de saída.
    """
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    epocas = range(1, len(historico_loss_treino) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Curvas de Treino", fontsize=14)

    ax1.plot(epocas, historico_loss_treino, label="Treino", color="royalblue")
    ax1.plot(epocas, historico_loss_val, label="Validação", color="tomato")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epocas, historico_acc_treino, label="Treino", color="royalblue")
    ax2.plot(epocas, historico_acc_val, label="Validação", color="tomato")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Acurácia")
    ax2.set_title("Acurácia")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plotar_matriz_confusao(
    matriz: np.ndarray,
    nomes_classes: list[str],
    caminho_saida: Path,
    normalizada: bool = True,
    dpi: int = 150,
) -> None:
    """Salva matriz de confusão como heatmap.

    Args:
        normalizada: Se True, normaliza por linha (acurácia por classe).
    """
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    if normalizada:
        total = matriz.sum(axis=1, keepdims=True)
        total = np.where(total == 0, 1, total)
        dados = matriz.astype(float) / total
        fmt = ".2f"
        titulo = "Matriz de Confusão (Normalizada)"
    else:
        dados = matriz
        fmt = "d"
        titulo = "Matriz de Confusão"

    fig, ax = plt.subplots(figsize=(max(8, len(nomes_classes)), max(6, len(nomes_classes) - 1)))
    sns.heatmap(
        dados,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=nomes_classes,
        yticklabels=nomes_classes,
        ax=ax,
    )
    ax.set_title(titulo)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plotar_distribuicao_classes(
    rotulos: np.ndarray,
    nomes_classes: list[str],
    caminho_saida: Path,
    titulo: str = "Distribuição de Classes",
    dpi: int = 150,
) -> None:
    """Salva gráfico de barras com a contagem por classe."""
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    contagens = np.bincount(rotulos, minlength=len(nomes_classes))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(nomes_classes)), contagens, color="steelblue")
    ax.set_xticks(range(len(nomes_classes)))
    ax.set_xticklabels(nomes_classes, rotation=45, ha="right")
    ax.set_ylabel("Quantidade")
    ax.set_title(titulo)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plotar_sobreposicao_xai(
    imagem_original: np.ndarray,
    mapa_xai: np.ndarray,
    caminho_saida: Path,
    titulo: str = "XAI",
    alfa: float = 0.5,
    dpi: int = 150,
) -> None:
    """Sobrepõe mapa de calor XAI sobre a imagem original.

    Args:
        imagem_original: Array (H, W, 3) uint8 ou float [0,1].
        mapa_xai: Array (H, W) float normalizado [0,1].
        alfa: Transparência do mapa de calor.
    """
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    img = imagem_original.copy()
    if img.dtype != np.uint8:
        img = (img * 255).clip(0, 255).astype(np.uint8)

    mapa_norm = (mapa_xai - mapa_xai.min()) / (mapa_xai.max() - mapa_xai.min() + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mapa_norm, cmap="jet")
    axes[1].set_title("Mapa XAI")
    axes[1].axis("off")

    axes[2].imshow(img)
    axes[2].imshow(mapa_norm, cmap="jet", alpha=alfa)
    axes[2].set_title("Sobreposição")
    axes[2].axis("off")

    fig.suptitle(titulo)
    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plotar_comparativo_modelos(
    resultados: dict[str, ResultadoAvaliacao],
    caminho_saida: Path,
    dpi: int = 150,
) -> None:
    """Gráfico de barras horizontais comparando acurácia de todos os modelos."""
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    nomes = list(resultados.keys())
    acuracias = [resultados[n].acuracia for n in nomes]
    f1s = [resultados[n].f1_macro for n in nomes]

    y = np.arange(len(nomes))
    altura = 0.35

    fig, ax = plt.subplots(figsize=(10, max(4, len(nomes) * 0.8)))
    ax.barh(y + altura / 2, acuracias, altura, label="Acurácia Top-1", color="steelblue")
    ax.barh(y - altura / 2, f1s, altura, label="F1 Macro", color="coral")
    ax.set_yticks(y)
    ax.set_yticklabels(nomes)
    ax.set_xlabel("Score")
    ax.set_title("Comparativo de Modelos")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, 1.05)

    # Anotações de valor
    for i, (acc, f1) in enumerate(zip(acuracias, f1s)):
        ax.text(acc + 0.005, i + altura / 2, f"{acc:.3f}", va="center", fontsize=8)
        ax.text(f1 + 0.005, i - altura / 2, f"{f1:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(caminho_saida, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
