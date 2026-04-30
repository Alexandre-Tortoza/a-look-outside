"""Cálculo e formatação de métricas de avaliação."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

CLASSES_GALAXY10 = [
    "Disk, Face-on, No Bulge",
    "Smooth, Completely Round",
    "Smooth, In-between Round",
    "Smooth, Cigar Shaped",
    "Disk, Edge-on, Boxy Bulge",
    "Disk, Edge-on, No Bulge",
    "Disk, Edge-on, Rounded Bulge",
    "Disk, Face-on, Tight Spiral",
    "Disk, Face-on, Medium Spiral",
    "Disk, Face-on, Loose Spiral",
]


@dataclass
class ResultadoAvaliacao:
    acuracia: float
    acuracia_top5: Optional[float]
    precisao_macro: float
    recall_macro: float
    f1_macro: float
    matriz_confusao: np.ndarray
    acuracia_por_classe: dict[int, float]
    relatorio: str
    nome_modelo: str = ""
    nome_experimento: str = ""
    nomes_classes: list[str] = field(default_factory=lambda: list(CLASSES_GALAXY10))


def calcular_metricas(
    y_verdadeiro: np.ndarray,
    y_previsto: np.ndarray,
    y_logits: Optional[np.ndarray] = None,
    nomes_classes: Optional[list[str]] = None,
    nome_modelo: str = "",
    nome_experimento: str = "",
) -> ResultadoAvaliacao:
    """Calcula métricas completas de classificação.

    Args:
        y_verdadeiro: Rótulos reais, shape (N,).
        y_previsto: Predições do modelo, shape (N,).
        y_logits: Logits/probabilidades, shape (N, C). Necessário para top-5.
        nomes_classes: Lista de nomes das classes.
        nome_modelo: Identificador do modelo para o resultado.
        nome_experimento: Identificador do experimento.

    Returns:
        ResultadoAvaliacao com todas as métricas computadas.
    """
    if nomes_classes is None:
        nomes_classes = list(CLASSES_GALAXY10)

    acuracia = float(accuracy_score(y_verdadeiro, y_previsto))
    precisao = float(precision_score(y_verdadeiro, y_previsto, average="macro", zero_division=0))
    recall = float(recall_score(y_verdadeiro, y_previsto, average="macro", zero_division=0))
    f1 = float(f1_score(y_verdadeiro, y_previsto, average="macro", zero_division=0))
    matriz = confusion_matrix(y_verdadeiro, y_previsto)
    relatorio = classification_report(
        y_verdadeiro, y_previsto, target_names=nomes_classes, zero_division=0
    )

    # Acurácia por classe a partir da diagonal da matriz de confusão
    acuracia_por_classe: dict[int, float] = {}
    for i in range(len(matriz)):
        total_classe = matriz[i].sum()
        acuracia_por_classe[i] = float(matriz[i, i] / total_classe) if total_classe > 0 else 0.0

    # Top-5
    top5: Optional[float] = None
    if y_logits is not None:
        n_classes = y_logits.shape[1]
        k = min(5, n_classes)
        top_k = np.argsort(y_logits, axis=1)[:, -k:]
        acertos = sum(int(y_verdadeiro[i] in top_k[i]) for i in range(len(y_verdadeiro)))
        top5 = acertos / len(y_verdadeiro)

    return ResultadoAvaliacao(
        acuracia=acuracia,
        acuracia_top5=top5,
        precisao_macro=precisao,
        recall_macro=recall,
        f1_macro=f1,
        matriz_confusao=matriz,
        acuracia_por_classe=acuracia_por_classe,
        relatorio=relatorio,
        nome_modelo=nome_modelo,
        nome_experimento=nome_experimento,
        nomes_classes=nomes_classes,
    )


def formatar_para_markdown(resultado: ResultadoAvaliacao) -> str:
    """Gera um relatório Markdown com todas as métricas.

    Returns:
        String Markdown pronta para salvar em resultados.md.
    """
    top5_str = f"{resultado.acuracia_top5:.4f}" if resultado.acuracia_top5 is not None else "N/A"

    linhas_por_classe = "\n".join(
        f"| {i} | {resultado.nomes_classes[i] if i < len(resultado.nomes_classes) else str(i)} "
        f"| {v:.4f} |"
        for i, v in sorted(resultado.acuracia_por_classe.items())
    )

    return f"""# Resultados — {resultado.nome_modelo}

**Experimento:** {resultado.nome_experimento}

## Métricas Globais

| Métrica | Valor |
|---------|-------|
| Acurácia Top-1 | {resultado.acuracia:.4f} |
| Acurácia Top-5 | {top5_str} |
| Precisão Macro | {resultado.precisao_macro:.4f} |
| Recall Macro | {resultado.recall_macro:.4f} |
| F1 Macro | {resultado.f1_macro:.4f} |

## Acurácia por Classe

| ID | Classe | Acurácia |
|----|--------|----------|
{linhas_por_classe}

## Relatório Completo

```
{resultado.relatorio}
```
"""
