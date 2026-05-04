"""Pipeline de benchmark sistematico VGG16 x Datasets.

Objetivo
--------
Determinar o melhor dataset (e tratamento) para treino do VGG16 e avaliar
generalizacao cross-dataset. O fine-tuning do melhor modelo no dataset
combinado busca empurrar o estado da arte.

Matriz de experimentos
----------------------
5 modelos x 3 datasets de avaliacao = 15 pontos de dados

                  Eval SDSS        Eval DECaLS      Eval Fusao
  T1 SDSS/raw    [in-dist split]  [cross full]     [cross full]
  T2 SDSS/hibr   [in-dist split]  [cross full]     [cross full]
  T3 DECaLS/raw  [cross full]     [in-dist split]  [cross full]
  T4 Fusao       [cross full]     [cross full]     [in-dist split]
  T5 FT→Fusao    [cross full]     [cross full]     [in-dist split]

Benchmarks individuais (15 + 2 de fine-tuning treino = 17 entradas)
  t1  VGG16 treina SDSS/raw    → val SDSS/raw (test split)
  t2  VGG16 treina SDSS/hibr   → val SDSS/hibr (test split)   [ja existe: 96.70%]
  t3  VGG16 treina DECaLS/raw  → val DECaLS/raw (test split)
  t4  VGG16 treina fusao       → val fusao (test split)
  t5  Fine-tune T2→fusao       → val fusao (test split)        [estado da arte]
  c1  T1 checkpoint            → test DECaLS (completo)
  c2  T1 checkpoint            → test fusao  (completo)
  c3  T2 checkpoint            → test DECaLS (completo)
  c4  T2 checkpoint            → test fusao  (completo)
  c5  T3 checkpoint            → test SDSS   (completo)
  c6  T3 checkpoint            → test fusao  (completo)
  c7  T4 checkpoint            → test SDSS   (completo)
  c8  T4 checkpoint            → test DECaLS (completo)
  c9  T5 checkpoint            → test SDSS   (completo)
  c10 T5 checkpoint            → test DECaLS (completo)

Uso
---
Listar todos os benchmarks sem rodar:
    python -m pipeline.vgg_datasets --listar

Rodar tudo (treino + avaliacao + relatorio):
    python -m pipeline.vgg_datasets

Pular treinamentos ja feitos e ir direto para avaliacao e relatorio:
    python -m pipeline.vgg_datasets --fases 2 3

Passar o melhor checkpoint manualmente para o fine-tuning:
    python -m pipeline.vgg_datasets --checkpoint-ft pesos/vgg16/vgg16_sdss_hibrido_ep50_20260502_221730.pth

Rodar apenas benchmarks especificos:
    python -m pipeline.vgg_datasets --ids t2 c3 c4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_RAIZ = Path(__file__).resolve().parent.parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

from utils.logger import configurar_logger_global, obter_logger
from utils.reproducibilidade import fixar_semente

# ---------------------------------------------------------------------------
# Definicao dos benchmarks
# ---------------------------------------------------------------------------

@dataclass
class BenchTreino:
    """Configuracao de um treino (produz um checkpoint reutilizavel)."""
    id: str
    descricao: str
    dataset: str
    versao: str
    tipo: str = "two_stage"     # "two_stage" | "finetuning"
    checkpoint_origem: Optional[str] = None  # so para tipo=finetuning


@dataclass
class BenchAvaliacao:
    """Configuracao de uma avaliacao (usa um checkpoint existente)."""
    id: str
    descricao: str
    treino_ref: str              # id do BenchTreino cujo checkpoint usar
    dataset_eval: str
    modo: str                    # "split" (test split) | "full" (dataset completo)
    versao_eval: str = "raw"


# Treinos — ordem de execucao
TREINOS: list[BenchTreino] = [
    BenchTreino("t1", "VGG16 treina SDSS/raw",       "sdss",   "raw"),
    BenchTreino("t2", "VGG16 treina SDSS/hibrido",   "sdss",   "hibrido"),
    BenchTreino("t3", "VGG16 treina DECaLS/raw",     "decals", "raw"),
    BenchTreino("t4", "VGG16 treina fusao",           "fusao",  "raw"),
    BenchTreino("t5", "Fine-tune T2 → fusao (estado da arte)",
                "fusao", "raw", tipo="finetuning", checkpoint_origem="t2"),
]

# Avaliacoes — matriz completa
AVALIACOES: list[BenchAvaliacao] = [
    # In-distribution (test split)
    BenchAvaliacao("t1",  "T1 val SDSS/raw",        "t1", "sdss",   "split"),
    BenchAvaliacao("t2",  "T2 val SDSS/hibrido",    "t2", "sdss",   "split", versao_eval="hibrido"),
    BenchAvaliacao("t3",  "T3 val DECaLS/raw",      "t3", "decals", "split"),
    BenchAvaliacao("t4",  "T4 val fusao",            "t4", "fusao",  "split"),
    BenchAvaliacao("t5",  "T5 val fusao (FT)",       "t5", "fusao",  "split"),

    # Cross-dataset — T1 (SDSS/raw)
    BenchAvaliacao("c1",  "T1 → DECaLS completo",   "t1", "decals", "full"),
    BenchAvaliacao("c2",  "T1 → fusao completo",     "t1", "fusao",  "full"),

    # Cross-dataset — T2 (SDSS/hibrido, melhor baseline)
    BenchAvaliacao("c3",  "T2 → DECaLS completo",   "t2", "decals", "full"),
    BenchAvaliacao("c4",  "T2 → fusao completo",     "t2", "fusao",  "full"),

    # Cross-dataset — T3 (DECaLS/raw)
    BenchAvaliacao("c5",  "T3 → SDSS completo",     "t3", "sdss",   "full"),
    BenchAvaliacao("c6",  "T3 → fusao completo",     "t3", "fusao",  "full"),

    # Cross-dataset — T4 (fusao)
    BenchAvaliacao("c7",  "T4 → SDSS completo",     "t4", "sdss",   "full"),
    BenchAvaliacao("c8",  "T4 → DECaLS completo",   "t4", "decals", "full"),

    # Cross-dataset — T5 (Fine-tuned, estado da arte)
    BenchAvaliacao("c9",  "T5 → SDSS completo",     "t5", "sdss",   "full"),
    BenchAvaliacao("c10", "T5 → DECaLS completo",   "t5", "decals", "full"),
]

# ---------------------------------------------------------------------------
# Registro de checkpoints (em memoria durante a execucao)
# ---------------------------------------------------------------------------

_checkpoints: dict[str, Path] = {}  # treino_id → caminho .pth


# ---------------------------------------------------------------------------
# Fase 0 — pre-requisitos
# ---------------------------------------------------------------------------

def fase_prereqs(log: logging.Logger) -> bool:
    """Cria o dataset fusao se ainda nao existir. Retorna True se OK."""
    from pre_processamento.fusao_datasets import criar_dataset_fusao, CAMINHO_FUSAO

    if not CAMINHO_FUSAO.exists():
        log.info("Dataset fusao nao existe — criando (pode demorar alguns minutos)...")
        criar_dataset_fusao()
        log.info("Fusao criada: %s", CAMINHO_FUSAO)
    else:
        mb = CAMINHO_FUSAO.stat().st_size / 1e6
        log.info("Dataset fusao OK: %s (%.1f MB)", CAMINHO_FUSAO, mb)
    return True


# ---------------------------------------------------------------------------
# Fase 1 — treinos
# ---------------------------------------------------------------------------

def fase_treinos(ids_filtro: Optional[list[str]], log: logging.Logger) -> None:
    """Executa os treinos necessarios, reutilizando checkpoints existentes."""

    for bench in TREINOS:
        if ids_filtro and bench.id not in ids_filtro:
            # Ainda tenta detectar checkpoint existente para uso nas avaliacoes
            _tentar_carregar_checkpoint(bench, log)
            continue

        log.info("")
        log.info("┌─ TREINO %s: %s", bench.id.upper(), bench.descricao)

        # Detecta se ja existe checkpoint para este (dataset, versao)
        ckpt_existente = _encontrar_checkpoint_existente("vgg16", bench.dataset, bench.versao)
        if ckpt_existente:
            log.info("│  Checkpoint existente detectado: %s", ckpt_existente.name)
            log.info("│  Pulando treino — usando checkpoint ja disponivel.")
            _checkpoints[bench.id] = ckpt_existente
            log.info("└─ OK")
            continue

        if bench.tipo == "finetuning":
            _executar_finetuning(bench, log)
        else:
            _executar_treino(bench, log)


def _executar_treino(bench: BenchTreino, log: logging.Logger) -> None:
    from modelos.vgg16.treino import treinar

    log.info("│  Treinando VGG16 em %s/%s...", bench.dataset, bench.versao)
    historico = treinar(config_override={
        "dataset": bench.dataset,
        "versao_dataset": bench.versao,
    })
    log.info("│  %s", historico.resumo())

    ckpt = _checkpoint_recente("vgg16")
    if ckpt:
        _checkpoints[bench.id] = ckpt
        log.info("└─ Checkpoint: %s", ckpt.name)
    else:
        log.error("└─ ERRO: checkpoint nao encontrado apos treino.")


def _executar_finetuning(bench: BenchTreino, log: logging.Logger) -> None:
    from modelos.vgg16.finetuning import fine_tuning

    origem_id = bench.checkpoint_origem
    ckpt_origem = _checkpoints.get(origem_id)
    if ckpt_origem is None:
        ckpt_origem = _encontrar_checkpoint_melhor("vgg16")
    if ckpt_origem is None:
        log.error("└─ ERRO: checkpoint de origem nao encontrado para fine-tuning.")
        return

    log.info("│  Fine-tuning a partir de: %s", ckpt_origem.name)
    log.info("│  Dataset: %s/%s", bench.dataset, bench.versao)

    historico = fine_tuning(ckpt_origem, config_override={
        "dataset": bench.dataset,
        "versao_dataset": bench.versao,
    })
    log.info("│  %s", historico.resumo())

    ckpt = _checkpoint_recente("vgg16")
    if ckpt:
        _checkpoints[bench.id] = ckpt
        log.info("└─ Checkpoint: %s", ckpt.name)
    else:
        log.error("└─ ERRO: checkpoint nao encontrado apos fine-tuning.")


def _tentar_carregar_checkpoint(bench: BenchTreino, log: logging.Logger) -> None:
    """Tenta popular _checkpoints com checkpoint existente sem treinar."""
    ckpt = _encontrar_checkpoint_existente("vgg16", bench.dataset, bench.versao)
    if ckpt:
        _checkpoints[bench.id] = ckpt
        log.info("Checkpoint pre-existente para %s: %s", bench.id, ckpt.name)


# ---------------------------------------------------------------------------
# Fase 2 — avaliacoes
# ---------------------------------------------------------------------------

def fase_avaliacoes(ids_filtro: Optional[list[str]], log: logging.Logger) -> dict[str, dict]:
    """Roda as avaliacoes e retorna resultados indexados por bench_id."""
    from modelos.vgg16.avaliacao import avaliar
    from utils.metricas import formatar_para_markdown

    dir_bench = _RAIZ / "docs" / "vgg16" / "benchmarks"
    dir_bench.mkdir(parents=True, exist_ok=True)

    resultados: dict[str, dict] = {}

    for bench in AVALIACOES:
        bench_id = f"{bench.treino_ref}_{bench.id}"

        if ids_filtro:
            # Filtra por treino_ref ou id de avaliacao
            if bench.treino_ref not in ids_filtro and bench.id not in ids_filtro:
                continue

        ckpt = _checkpoints.get(bench.treino_ref)
        if ckpt is None:
            ckpt = _encontrar_checkpoint_existente("vgg16", _dataset_do_treino(bench.treino_ref),
                                                   _versao_do_treino(bench.treino_ref))
        if ckpt is None:
            log.warning("Pulando avaliacao %s: checkpoint %s nao disponivel.",
                        bench_id, bench.treino_ref)
            continue

        log.info("Avaliando %-6s | %-35s | modo=%-5s | eval=%s",
                 bench_id, bench.descricao, bench.modo, bench.dataset_eval)

        try:
            if bench.modo == "full":
                resultado = avaliar(ckpt, config_override={"dataset_teste": bench.dataset_eval})
            else:  # split
                resultado = avaliar(ckpt, config_override={
                    "dataset":          bench.dataset_eval,
                    "versao_dataset":   bench.versao_eval,
                })

            resultados[bench_id] = {
                "treino":    bench.treino_ref,
                "eval":      bench.dataset_eval,
                "modo":      bench.modo,
                "acuracia":  resultado.acuracia,
                "f1_macro":  resultado.f1_macro,
                "precisao":  resultado.precisao_macro,
                "recall":    resultado.recall_macro,
                "descricao": bench.descricao,
            }

            md = formatar_para_markdown(resultado)
            saida = dir_bench / f"{bench_id}.md"
            saida.write_text(md, encoding="utf-8")
            log.info("  → acc=%.4f  f1=%.4f  [%s]", resultado.acuracia, resultado.f1_macro, saida.name)

        except Exception as exc:
            log.error("  ERRO em %s: %s", bench_id, exc)

    return resultados


def _dataset_do_treino(treino_id: str) -> str:
    for t in TREINOS:
        if t.id == treino_id:
            return t.dataset
    return "sdss"


def _versao_do_treino(treino_id: str) -> str:
    for t in TREINOS:
        if t.id == treino_id:
            return t.versao
    return "raw"


# ---------------------------------------------------------------------------
# Fase 3 — relatorio
# ---------------------------------------------------------------------------

def fase_relatorio(resultados: dict[str, dict], log: logging.Logger) -> Path:
    """Gera relatorio markdown completo com a matriz de benchmarks."""
    dir_bench = _RAIZ / "docs" / "vgg16" / "benchmarks"
    dir_bench.mkdir(parents=True, exist_ok=True)

    linhas: list[str] = []
    a = linhas.append

    a("# Benchmark VGG16 × Datasets\n\n")
    a("**Modelos avaliados:** T1 (SDSS/raw), T2 (SDSS/hibrido), T3 (DECaLS/raw), "
      "T4 (fusao), T5 (fine-tune T2→fusao)  \n")
    a("**Datasets de avaliacao:** SDSS, DECaLS, fusao  \n")
    a("**Modo split:** test split 15% (in-distribution)  \n")
    a("**Modo full:** dataset completo (cross-dataset generalization)  \n\n")

    # ── Tabela de treinamentos ─────────────────────────────────────────────
    a("## Treinos (in-distribution — val_acc no test split)\n\n")
    a("| ID | Descricao | Dataset treino | val_acc | F1 macro |\n")
    a("|---|---|---|---:|---:|\n")
    for bench in TREINOS:
        chave = f"{bench.id}_{bench.id}"
        r = resultados.get(chave)
        if r:
            a(f"| **{bench.id}** | {bench.descricao} | {bench.dataset}/{bench.versao} "
              f"| **{r['acuracia']:.4f}** | {r['f1_macro']:.4f} |\n")
        else:
            a(f"| **{bench.id}** | {bench.descricao} | {bench.dataset}/{bench.versao} "
              f"| — | — |\n")

    # ── Matriz cross-dataset ───────────────────────────────────────────────
    a("\n## Matriz Cross-Dataset (acuracia)\n\n")
    a("> Diagonal = in-distribution (test split). Fora = cross-dataset (dataset completo).\n\n")

    datasets_eval = ["sdss", "decals", "fusao"]
    header = "| Treino \\ Eval |" + "".join(f" **{d.upper()}** |" for d in datasets_eval)
    sep    = "|---|" + "---:|" * len(datasets_eval)
    a(header + "\n")
    a(sep + "\n")

    _AVALIACAO_KEY = {
        # (treino_id, dataset_eval) → bench_id de avaliacao
        ("t1", "sdss"):   "t1_t1",
        ("t1", "decals"): "t1_c1",
        ("t1", "fusao"):  "t1_c2",
        ("t2", "sdss"):   "t2_t2",
        ("t2", "decals"): "t2_c3",
        ("t2", "fusao"):  "t2_c4",
        ("t3", "sdss"):   "t3_c5",
        ("t3", "decals"): "t3_t3",
        ("t3", "fusao"):  "t3_c6",
        ("t4", "sdss"):   "t4_c7",
        ("t4", "decals"): "t4_c8",
        ("t4", "fusao"):  "t4_t4",
        ("t5", "sdss"):   "t5_c9",
        ("t5", "decals"): "t5_c10",
        ("t5", "fusao"):  "t5_t5",
    }

    for treino in TREINOS:
        desc_curta = {
            "t1": "SDSS/raw",
            "t2": "SDSS/hibrido ★",
            "t3": "DECaLS/raw",
            "t4": "fusao",
            "t5": "FT→fusao ⚡",
        }.get(treino.id, treino.id)

        linha = f"| **{desc_curta}** |"
        for ds in datasets_eval:
            chave = _AVALIACAO_KEY.get((treino.id, ds))
            r = resultados.get(chave) if chave else None
            eh_diagonal = (
                (treino.id == "t1" and ds == "sdss") or
                (treino.id == "t2" and ds == "sdss") or
                (treino.id == "t3" and ds == "decals") or
                (treino.id in ("t4", "t5") and ds == "fusao")
            )
            if r:
                val = f"{r['acuracia']:.4f}"
                if eh_diagonal:
                    val = f"**{val}**"
                linha += f" {val} |"
            else:
                linha += " — |"
        a(linha + "\n")

    # ── Analise de generalizacao ──────────────────────────────────────────
    a("\n## Analise de Generalizacao (gap in-dist vs cross)\n\n")
    a("| Treino | In-dist acc | Melhor cross acc | Gap |\n")
    a("|---|---:|---:|---:|\n")

    _IN_DIST = {"t1": "t1_t1", "t2": "t2_t2", "t3": "t3_t3", "t4": "t4_t4", "t5": "t5_t5"}
    _CROSS = {
        "t1": ["t1_c1", "t1_c2"],
        "t2": ["t2_c3", "t2_c4"],
        "t3": ["t3_c5", "t3_c6"],
        "t4": ["t4_c7", "t4_c8"],
        "t5": ["t5_c9", "t5_c10"],
    }

    for treino in TREINOS:
        r_ind = resultados.get(_IN_DIST[treino.id])
        crosses = [resultados[k] for k in _CROSS[treino.id] if k in resultados]
        if r_ind and crosses:
            melhor_cross = max(c["acuracia"] for c in crosses)
            gap = r_ind["acuracia"] - melhor_cross
            a(f"| {treino.id} ({treino.dataset}/{treino.versao}) "
              f"| {r_ind['acuracia']:.4f} | {melhor_cross:.4f} | {gap:+.4f} |\n")

    # ── Impacto do fine-tuning ────────────────────────────────────────────
    a("\n## Impacto do Fine-Tuning (T5 vs T2)\n\n")
    a("| Avaliacao | T2 (SDSS/hibrido) | T5 (FT→fusao) | Delta |\n")
    a("|---|---:|---:|---:|\n")

    pares = [
        ("SDSS completo",   "t2_t2",  "t5_c9"),
        ("DECaLS completo", "t2_c3",  "t5_c10"),
        ("fusao test split","t2_c4",  "t5_t5"),
    ]
    for label, k_t2, k_t5 in pares:
        r2 = resultados.get(k_t2)
        r5 = resultados.get(k_t5)
        if r2 and r5:
            delta = r5["acuracia"] - r2["acuracia"]
            sinal = "+" if delta >= 0 else ""
            a(f"| {label} | {r2['acuracia']:.4f} | {r5['acuracia']:.4f} | **{sinal}{delta:.4f}** |\n")

    # ── Conclusao ─────────────────────────────────────────────────────────
    a("\n## Conclusao\n\n")

    # Melhor dataset de treino (in-distribution)
    best_ind = max(
        ((tid, resultados.get(_IN_DIST[tid])) for tid in _IN_DIST),
        key=lambda x: x[1]["acuracia"] if x[1] else -1,
        default=("?", None),
    )
    if best_ind[1]:
        a(f"- **Melhor dataset in-distribution:** `{best_ind[0]}` "
          f"(acc={best_ind[1]['acuracia']:.4f})\n")

    # Melhor generalizacao cross
    cross_flat = []
    for tid, keys in _CROSS.items():
        for k in keys:
            if k in resultados:
                cross_flat.append((tid, k, resultados[k]["acuracia"]))
    if cross_flat:
        best_cross = max(cross_flat, key=lambda x: x[2])
        a(f"- **Melhor cross-dataset:** treinado em `{best_cross[0]}`, "
          f"avaliado como `{best_cross[1]}` (acc={best_cross[2]:.4f})\n")

    a("- Detalhes por benchmark: `docs/vgg16/benchmarks/*.md`\n")

    # Salvar JSON com todos os resultados
    json_saida = dir_bench / "resultados.json"
    json_saida.write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    saida = dir_bench / "relatorio.md"
    saida.write_text("".join(linhas), encoding="utf-8")
    log.info("Relatorio salvo em: %s", saida)
    log.info("Resultados JSON: %s", json_saida)
    return saida


# ---------------------------------------------------------------------------
# Utilitarios de checkpoint
# ---------------------------------------------------------------------------

def _encontrar_checkpoint_existente(modelo: str, dataset: str, versao: str) -> Optional[Path]:
    """Busca o .pth mais recente com (dataset, versao_dataset) nos JSONs."""
    dir_pesos = _RAIZ / "pesos" / modelo
    if not dir_pesos.exists():
        return None

    candidatos: list[tuple[float, Path]] = []
    for arq_json in dir_pesos.glob("*.json"):
        try:
            dados = json.loads(arq_json.read_text(encoding="utf-8"))
            params = dados.get("hiperparametros", {})
            if params.get("dataset") == dataset and params.get("versao_dataset") == versao:
                pth = arq_json.with_suffix(".pth")
                if pth.exists():
                    candidatos.append((arq_json.stat().st_mtime, pth))
        except Exception:
            continue

    if not candidatos:
        return None
    candidatos.sort(reverse=True)
    return candidatos[0][1]


def _encontrar_checkpoint_melhor(modelo: str) -> Optional[Path]:
    """Retorna o .pth com maior melhor_val_acc em todos os JSONs do modelo."""
    dir_pesos = _RAIZ / "pesos" / modelo
    if not dir_pesos.exists():
        return None

    melhor_acc = -1.0
    melhor_pth: Optional[Path] = None
    for arq_json in dir_pesos.glob("*.json"):
        try:
            dados = json.loads(arq_json.read_text(encoding="utf-8"))
            acc = dados.get("metricas", {}).get("melhor_val_acc", -1.0)
            pth = arq_json.with_suffix(".pth")
            if acc > melhor_acc and pth.exists():
                melhor_acc = acc
                melhor_pth = pth
        except Exception:
            continue
    return melhor_pth


def _checkpoint_recente(modelo: str) -> Optional[Path]:
    """Retorna o .pth mais recente (por mtime) do modelo."""
    dir_pesos = _RAIZ / "pesos" / modelo
    if not dir_pesos.exists():
        return None
    arquivos = sorted(dir_pesos.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return arquivos[0] if arquivos else None


def _pre_popular_checkpoints(log: logging.Logger) -> None:
    """Tenta popular _checkpoints com o que ja existe em disco."""
    for bench in TREINOS:
        if bench.id not in _checkpoints:
            if bench.tipo == "finetuning":
                continue  # nao ha como detectar checkpoint de FT sem contexto
            ckpt = _encontrar_checkpoint_existente("vgg16", bench.dataset, bench.versao)
            if ckpt:
                _checkpoints[bench.id] = ckpt
                log.debug("Pre-popular: %s → %s", bench.id, ckpt.name)


# ---------------------------------------------------------------------------
# Modo --listar
# ---------------------------------------------------------------------------

def listar_benchmarks() -> None:
    """Imprime todos os benchmarks sem executar nada."""
    print("\n" + "=" * 72)
    print("  BENCHMARKS VGG16 × DATASETS")
    print("=" * 72)

    print("\n── TREINOS (Fase 1) ──────────────────────────────────────────────")
    print(f"  {'ID':<6} {'Tipo':<12} {'Dataset/Versao':<20} {'Descricao'}")
    print(f"  {'─'*6} {'─'*12} {'─'*20} {'─'*32}")
    for t in TREINOS:
        print(f"  {t.id:<6} {t.tipo:<12} {t.dataset+'/'+t.versao:<20} {t.descricao}")

    print("\n── AVALIACOES (Fase 2) — 15 pontos na matriz ─────────────────────")
    print(f"  {'ID':<10} {'Modo':<6} {'Treino':<6} {'Eval dataset':<12} {'Descricao'}")
    print(f"  {'─'*10} {'─'*6} {'─'*6} {'─'*12} {'─'*35}")
    for a in AVALIACOES:
        bench_id = f"{a.treino_ref}_{a.id}"
        print(f"  {bench_id:<10} {a.modo:<6} {a.treino_ref:<6} {a.dataset_eval:<12} {a.descricao}")

    print("\n── MATRIZ COMPLETA ───────────────────────────────────────────────")
    header = f"  {'Treino':<18} {'SDSS':>10} {'DECaLS':>10} {'fusao':>10}"
    print(header)
    print(f"  {'─'*18} {'─'*10} {'─'*10} {'─'*10}")
    matriz = {
        "t1 SDSS/raw":       ("in-dist", "cross",   "cross"),
        "t2 SDSS/hibrido ★": ("in-dist", "cross",   "cross"),
        "t3 DECaLS/raw":     ("cross",   "in-dist", "cross"),
        "t4 fusao":          ("cross",   "cross",   "in-dist"),
        "t5 FT→fusao ⚡":   ("cross",   "cross",   "in-dist"),
    }
    for nome, (sdss, decals, fusao) in matriz.items():
        print(f"  {nome:<18} {sdss:>10} {decals:>10} {fusao:>10}")

    print()
    print("  in-dist = test split 15% do proprio dataset de treino")
    print("  cross   = dataset COMPLETO (generalizacao out-of-distribution)")
    print()
    print("── USO ────────────────────────────────────────────────────────────")
    print("  Tudo:          python -m pipeline.vgg_datasets")
    print("  So avaliacao:  python -m pipeline.vgg_datasets --fases 2 3")
    print("  Ids especif.:  python -m pipeline.vgg_datasets --ids t2 c3 c4")
    print("  Melhor ckpt:   python -m pipeline.vgg_datasets --checkpoint-ft pesos/vgg16/X.pth")
    print()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark sistematico VGG16 x Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--listar", action="store_true",
                        help="Listar todos os benchmarks sem executar")
    parser.add_argument(
        "--fases", nargs="+", type=int, default=[0, 1, 2, 3],
        choices=[0, 1, 2, 3], metavar="N",
        help="Fases a executar: 0=pre-req(fusao), 1=treinos, 2=avaliacoes, 3=relatorio",
    )
    parser.add_argument(
        "--ids", nargs="+", default=None,
        metavar="ID",
        help="IDs de treinos/avaliacoes especificos (ex: t2 c3 c4). Default: todos.",
    )
    parser.add_argument(
        "--checkpoint-ft", type=Path, default=None, dest="checkpoint_ft",
        help="Checkpoint a usar como origem do fine-tuning (t5). "
             "Padrao: automaticamente o de maior val_acc.",
    )
    args = parser.parse_args()

    if args.listar:
        listar_benchmarks()
        return

    configurar_logger_global(logging.INFO)
    log = obter_logger("vgg_datasets",
                       arquivo_log=_RAIZ / "docs" / "vgg_datasets.log")
    fixar_semente(42)

    log.info("Pipeline VGG16 × Datasets | fases=%s | ids=%s", args.fases, args.ids or "todas")

    # Injeta checkpoint-ft manualmente se fornecido
    if args.checkpoint_ft:
        if not args.checkpoint_ft.exists():
            log.error("--checkpoint-ft nao encontrado: %s", args.checkpoint_ft)
            sys.exit(1)
        _checkpoints["t2"] = args.checkpoint_ft
        log.info("Checkpoint FT (t2) fixado manualmente: %s", args.checkpoint_ft.name)

    # Tenta pre-popular com checkpoints ja existentes em disco
    _pre_popular_checkpoints(log)

    resultados: dict[str, dict] = {}

    if 0 in args.fases:
        fase_prereqs(log)

    if 1 in args.fases:
        fase_treinos(args.ids, log)

    if 2 in args.fases:
        resultados = fase_avaliacoes(args.ids, log)

    if 3 in args.fases:
        saida = fase_relatorio(resultados, log)
        print(f"\nRelatorio completo: {saida}")

    log.info("Pipeline concluido.")


if __name__ == "__main__":
    main()
