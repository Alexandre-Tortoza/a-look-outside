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


def plotar_grade_xai_por_classe(
    resultados_por_classe: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    dir_saida: Path,
    imagens_por_linha: int = 5,
    alfa: float = 0.5,
    dpi: int = 150,
) -> None:
    """Salva um PNG por classe com pares (original | XAI) em grade.

    Cada par mostra a imagem original à esquerda e o overlay XAI à direita.
    Os pares são separados por espaço visual maior que o espaço interno do par.

    Args:
        resultados_por_classe: {nome_classe: [(img_original, mapa_xai), ...]}
        dir_saida: Diretório onde os PNGs serão salvos (um por classe).
        imagens_por_linha: Número de pares por linha.
        alfa: Transparência do heatmap.
        dpi: Resolução do PNG — pode ser grande, boa para zoom.
    """
    import matplotlib.gridspec as gridspec

    dir_saida = Path(dir_saida)
    dir_saida.mkdir(parents=True, exist_ok=True)

    IMG_SZ = 2.8   # polegadas por imagem individual
    GAP_INNER = 0.04  # espaço interno do par (orig <-> xai), em fração
    GAP_OUTER = 0.25  # espaço entre pares

    for nome_classe, pares in resultados_por_classe.items():
        n = len(pares)
        if n == 0:
            continue

        n_linhas = max(1, (n + imagens_por_linha - 1) // imagens_por_linha)

        # Larguras das colunas: [1, 1, spacer, 1, 1, spacer, ...] sem spacer no fim
        col_widths: list[float] = []
        spacer_w = GAP_OUTER / (1 + GAP_OUTER)  # proporcional à largura de uma imagem
        for i in range(imagens_por_linha):
            col_widths.extend([1.0, 1.0])
            if i < imagens_por_linha - 1:
                col_widths.append(spacer_w)

        total_cols = len(col_widths)

        fig_largura = imagens_por_linha * 2 * IMG_SZ + (imagens_por_linha - 1) * GAP_OUTER * IMG_SZ + 0.6
        fig_altura = n_linhas * IMG_SZ + n_linhas * 0.5 + 0.8

        fig = plt.figure(figsize=(fig_largura, fig_altura))
        gs = gridspec.GridSpec(
            n_linhas, total_cols,
            figure=fig,
            width_ratios=col_widths,
            wspace=GAP_INNER,
            hspace=0.35,
            left=0.02, right=0.98,
            top=0.93, bottom=0.02,
        )

        def _col_orig(par_idx: int) -> int:
            """Índice da coluna 'original' dado o índice do par na linha."""
            return par_idx * 3 if par_idx < imagens_por_linha - 1 or imagens_por_linha > 1 else par_idx * 2

        def _col_xai(par_idx: int) -> int:
            return _col_orig(par_idx) + 1

        # Recalcular mapeamento correto: par i -> colunas baseadas em col_widths
        # col_widths: pares de [1,1] separados por [spacer]
        # par 0: cols 0,1; par 1: cols 3,4; par 2: cols 6,7 ...
        # ou sem spacer no fim: par N-1: cols (N-1)*3 e (N-1)*3+1

        def _par_para_colunas(par_idx: int) -> tuple[int, int]:
            return par_idx * 3, par_idx * 3 + 1

        for idx, (img, mapa) in enumerate(pares):
            row = idx // imagens_por_linha
            par_na_linha = idx % imagens_por_linha

            img_u8 = img.copy()
            if img_u8.dtype != np.uint8:
                img_u8 = (img_u8 * 255).clip(0, 255).astype(np.uint8)
            img_u8 = img_u8[..., :3]

            mapa_n = (mapa - mapa.min()) / (mapa.max() - mapa.min() + 1e-8)

            c_orig, c_xai = _par_para_colunas(par_na_linha)

            ax_orig = fig.add_subplot(gs[row, c_orig])
            ax_orig.imshow(img_u8)
            ax_orig.axis("off")
            if row == 0 and par_na_linha == 0:
                ax_orig.set_title("Original", fontsize=7, pad=2)

            ax_xai = fig.add_subplot(gs[row, c_xai])
            ax_xai.imshow(img_u8)
            ax_xai.imshow(mapa_n, cmap="jet", alpha=alfa)
            ax_xai.axis("off")
            if row == 0 and par_na_linha == 0:
                ax_xai.set_title("XAI", fontsize=7, pad=2)

        fig.suptitle(f"XAI — {nome_classe}", fontsize=11)
        caminho_png = dir_saida / f"{nome_classe}.png"
        plt.savefig(caminho_png, dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plotar_grade_xai(
    resultados_por_classe: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    caminho_saida: Path,
    imagens_por_linha: int = 10,
    alfa: float = 0.5,
    dpi: int = 100,
) -> None:
    """Grade de overlays XAI organizados por classe (uma linha por classe).

    Args:
        resultados_por_classe: {nome_classe: [(img_original, mapa_xai), ...]}
            img_original: (H, W, 3) uint8; mapa_xai: (H, W) float [0,1].
        caminho_saida: Caminho do PNG de saída.
        imagens_por_linha: Número de colunas na grade.
        alfa: Transparência do heatmap sobre a imagem.
        dpi: Resolução do PNG.
    """
    caminho_saida = Path(caminho_saida)
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)

    classes = list(resultados_por_classe.keys())
    total_linhas = sum(
        max(1, (len(resultados_por_classe[c]) + imagens_por_linha - 1) // imagens_por_linha)
        for c in classes
    )

    img_sz = 0.9
    fig_largura = imagens_por_linha * img_sz + 1.4
    fig_altura = max(total_linhas * img_sz + len(classes) * 0.1, 2.0)

    fig = plt.figure(figsize=(fig_largura, fig_altura))

    linha_atual = 0
    for nome_classe in classes:
        pares = resultados_por_classe[nome_classe]
        n = len(pares)
        n_sublinhas = max(1, (n + imagens_por_linha - 1) // imagens_por_linha)

        for sub in range(n_sublinhas):
            inicio = sub * imagens_por_linha
            fim = min(inicio + imagens_por_linha, n)

            for col, (img, mapa) in enumerate(pares[inicio:fim]):
                idx = (linha_atual + sub) * imagens_por_linha + col + 1
                ax = fig.add_subplot(total_linhas, imagens_por_linha, idx)

                img_u8 = img.copy()
                if img_u8.dtype != np.uint8:
                    img_u8 = (img_u8 * 255).clip(0, 255).astype(np.uint8)

                mapa_n = (mapa - mapa.min()) / (mapa.max() - mapa.min() + 1e-8)

                ax.imshow(img_u8[..., :3])
                ax.imshow(mapa_n, cmap="jet", alpha=alfa)
                ax.axis("off")

                if sub == 0 and col == 0:
                    ax.set_ylabel(nome_classe, fontsize=7, rotation=0,
                                  labelpad=42, va="center", ha="right")
                    ax.yaxis.set_label_position("left")

        linha_atual += n_sublinhas

    fig.suptitle("XAI — sobreposição por classe", fontsize=9, y=1.01)
    plt.tight_layout(pad=0.3)
    plt.savefig(caminho_saida, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
