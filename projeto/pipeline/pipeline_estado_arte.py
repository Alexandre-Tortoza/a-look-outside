"""Pipeline de experimentos cross-dataset e estado da arte com VGG16.

Etapas disponíveis
------------------
1  Cross SDSS → DECaLS   Treina VGG16 no SDSS e avalia no DECaLS completo
2  Cross DECaLS → SDSS   Treina VGG16 no DECaLS e avalia no SDSS completo
3  Criar fusao            Combina SDSS + DECaLS num unico H5 (224x224, so imagens)
4  Fine-tuning            Ajusta o melhor checkpoint VGG16 no dataset fusao
5  Avaliacao final        Avalia o modelo fine-tuned em SDSS, DECaLS e fusao

Uso
---
Rodar tudo (usa o melhor checkpoint VGG16 disponivel para etapas 4 e 5):
    python -m pipeline.pipeline_estado_arte

Rodar so etapas 3, 4 e 5 a partir de um checkpoint especifico:
    python -m pipeline.pipeline_estado_arte --etapas 3 4 5 \\
        --checkpoint pesos/vgg16/vgg16_sdss_hibrido_ep50_20260502_221730.pth

Rodar so avaliacao final com checkpoint existente:
    python -m pipeline.pipeline_estado_arte --etapas 5 \\
        --checkpoint pesos/vgg16/NOME.pth

Flags de dataset:
    --versao-sdss    Versao do SDSS usada na etapa 1 (padrao: hibrido)
    --versao-decals  Versao do DECaLS usada na etapa 2 (padrao: raw)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Garante que o projeto esta no PYTHONPATH ao rodar de qualquer diretório
_RAIZ = Path(__file__).resolve().parent.parent
if str(_RAIZ) not in sys.path:
    sys.path.insert(0, str(_RAIZ))

from utils.logger import configurar_logger_global, obter_logger
from utils.reproducibilidade import fixar_semente


# ---------------------------------------------------------------------------
# Etapa 1 / 2 — cross-dataset treino + avaliação
# ---------------------------------------------------------------------------

def etapa_cross(
    dataset_treino: str,
    versao_treino: str,
    dataset_avaliacao: str,
    log: logging.Logger,
) -> Path | None:
    """Treina VGG16 num dataset e avalia em outro.

    Returns:
        Caminho do checkpoint salvo, ou None em caso de falha.
    """
    from modelos.vgg16.treino import treinar

    log.info("=" * 70)
    log.info("CROSS: treino=%s/%s  avaliacao=%s", dataset_treino, versao_treino, dataset_avaliacao)
    log.info("=" * 70)

    historico = treinar(config_override={
        "dataset": dataset_treino,
        "versao_dataset": versao_treino,
    })
    log.info("Treino concluido: %s", historico.resumo())

    checkpoint = _checkpoint_recente("vgg16")
    if checkpoint is None:
        log.error("Nenhum checkpoint VGG16 encontrado apos treino.")
        return None

    log.info("Avaliando em %s (dataset completo)...", dataset_avaliacao)
    tag = f"cross_{dataset_treino}_{versao_treino}_para_{dataset_avaliacao}"
    _avaliar_cross(checkpoint, dataset_avaliacao, tag, log)
    return checkpoint


# ---------------------------------------------------------------------------
# Etapa 3 — criar dataset fusao
# ---------------------------------------------------------------------------

def etapa_criar_fusao(log: logging.Logger) -> Path:
    """Cria dataset/processados/fusao.h5 se ainda nao existir."""
    from pre_processamento.fusao_datasets import criar_dataset_fusao, CAMINHO_FUSAO

    log.info("=" * 70)
    log.info("FUSAO: criando dataset combinado SDSS + DECaLS")
    log.info("=" * 70)
    caminho = criar_dataset_fusao()
    log.info("Fusao pronta: %s", caminho)
    return caminho


# ---------------------------------------------------------------------------
# Etapa 4 — fine-tuning do melhor modelo no dataset fusao
# ---------------------------------------------------------------------------

def etapa_fine_tuning(checkpoint: Path, log: logging.Logger) -> Path | None:
    """Fine-tuning do VGG16 no dataset fusao a partir de um checkpoint."""
    from modelos.vgg16.finetuning import fine_tuning
    from pre_processamento.fusao_datasets import CAMINHO_FUSAO

    if not CAMINHO_FUSAO.exists():
        log.error("Dataset fusao nao encontrado em %s. Execute a etapa 3 primeiro.", CAMINHO_FUSAO)
        return None

    log.info("=" * 70)
    log.info("FINE-TUNING: VGG16 no dataset fusao")
    log.info("  Checkpoint de origem: %s", checkpoint)
    log.info("=" * 70)

    historico = fine_tuning(checkpoint, config_override={
        "dataset": "fusao",
        "versao_dataset": "raw",
    })
    log.info("Fine-tuning concluido: %s", historico.resumo())

    novo_checkpoint = _checkpoint_recente("vgg16")
    return novo_checkpoint


# ---------------------------------------------------------------------------
# Etapa 5 — avaliação final do modelo estado da arte
# ---------------------------------------------------------------------------

def etapa_avaliacao_final(checkpoint: Path, log: logging.Logger) -> None:
    """Avalia o modelo fine-tuned em SDSS, DECaLS e fusao."""
    from pre_processamento.fusao_datasets import CAMINHO_FUSAO

    log.info("=" * 70)
    log.info("AVALIACAO FINAL: modelo estado da arte")
    log.info("  Checkpoint: %s", checkpoint)
    log.info("=" * 70)

    # SDSS e DECaLS: avaliacao cross-dataset (dataset completo como teste)
    for ds in ("sdss", "decals"):
        log.info("--- Avaliando em %s (dataset completo) ---", ds.upper())
        _avaliar_cross(checkpoint, ds, f"estado_arte_{ds}", log)

    # Fusao: avaliacao no test split (evita leakage de treino)
    if CAMINHO_FUSAO.exists():
        log.info("--- Avaliando em FUSAO (test split 15%%) ---")
        _avaliar_split(checkpoint, "fusao", "estado_arte_fusao", log)
    else:
        log.warning("Dataset fusao nao encontrado — avaliacao em fusao ignorada.")


# ---------------------------------------------------------------------------
# Helpers de avaliação
# ---------------------------------------------------------------------------

def _avaliar_cross(
    checkpoint: Path,
    dataset_teste: str,
    tag: str,
    log: logging.Logger,
) -> None:
    """Avalia o modelo no dataset_teste completo (cross-dataset, sem split)."""
    from modelos.vgg16.avaliacao import avaliar
    from utils.metricas import formatar_para_markdown

    resultado = avaliar(checkpoint, config_override={"dataset_teste": dataset_teste})
    _salvar_resultado(resultado, tag, log)


def _avaliar_split(
    checkpoint: Path,
    dataset: str,
    tag: str,
    log: logging.Logger,
) -> None:
    """Avalia o modelo no test split do dataset (15% nao visto no treino)."""
    from modelos.vgg16.avaliacao import avaliar
    from utils.metricas import formatar_para_markdown

    resultado = avaliar(checkpoint, config_override={
        "dataset": dataset,
        "versao_dataset": "raw",
    })
    _salvar_resultado(resultado, tag, log)


def _salvar_resultado(resultado, tag: str, log: logging.Logger) -> None:
    from utils.metricas import formatar_para_markdown

    md = formatar_para_markdown(resultado)
    dir_docs = _RAIZ / "docs" / "vgg16"
    dir_docs.mkdir(parents=True, exist_ok=True)
    saida = dir_docs / f"resultados_{tag}.md"
    saida.write_text(md, encoding="utf-8")
    log.info("  Acuracia=%.4f | F1=%.4f | -> %s", resultado.acuracia, resultado.f1_macro, saida.name)


# ---------------------------------------------------------------------------
# Utilitario — melhor checkpoint por val_acc
# ---------------------------------------------------------------------------

def _checkpoint_recente(modelo: str) -> Path | None:
    """Retorna o .pth mais recente (por mtime) de um modelo."""
    dir_pesos = _RAIZ / "pesos" / modelo
    if not dir_pesos.exists():
        return None
    arquivos = sorted(dir_pesos.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return arquivos[0] if arquivos else None


def _checkpoint_melhor(modelo: str) -> Path | None:
    """Retorna o .pth com maior val_acc entre todos os JSONs de um modelo."""
    import json

    dir_pesos = _RAIZ / "pesos" / modelo
    if not dir_pesos.exists():
        return None

    melhor_acc = -1.0
    melhor_pth = None
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


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline estado da arte VGG16 — cross-dataset + fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--etapas",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        choices=[1, 2, 3, 4, 5],
        metavar="N",
        help=(
            "Etapas a executar (padrao: todas). "
            "1=cross SDSS→DECaLS, 2=cross DECaLS→SDSS, "
            "3=criar fusao, 4=fine-tuning, 5=avaliacao final"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint VGG16 para as etapas 4 e 5. "
            "Se omitido, usa o de maior val_acc disponivel."
        ),
    )
    parser.add_argument(
        "--versao-sdss",
        default="hibrido",
        dest="versao_sdss",
        help="Versao do SDSS para etapa 1 (padrao: hibrido)",
    )
    parser.add_argument(
        "--versao-decals",
        default="raw",
        dest="versao_decals",
        help="Versao do DECaLS para etapa 2 (padrao: raw)",
    )
    args = parser.parse_args()

    configurar_logger_global(logging.INFO)
    log = obter_logger("pipeline", arquivo_log=_RAIZ / "docs" / "pipeline_estado_arte.log")
    fixar_semente(42)

    log.info("Pipeline Estado da Arte — VGG16")
    log.info("Etapas: %s", args.etapas)

    # Checkpoint de trabalho — atualizado a cada etapa que treina/fine-tuna
    checkpoint_atual: Path | None = args.checkpoint

    # ── Etapa 1: cross SDSS → DECaLS ────────────────────────────────────────
    if 1 in args.etapas:
        resultado = etapa_cross("sdss", args.versao_sdss, "decals", log)
        if resultado and checkpoint_atual is None:
            checkpoint_atual = resultado

    # ── Etapa 2: cross DECaLS → SDSS ────────────────────────────────────────
    if 2 in args.etapas:
        etapa_cross("decals", args.versao_decals, "sdss", log)

    # ── Etapa 3: criar dataset fusao ─────────────────────────────────────────
    if 3 in args.etapas:
        etapa_criar_fusao(log)

    # ── Etapa 4: fine-tuning no fusao ────────────────────────────────────────
    if 4 in args.etapas:
        if checkpoint_atual is None:
            checkpoint_atual = _checkpoint_melhor("vgg16")
        if checkpoint_atual is None:
            log.error(
                "Etapa 4 requer um checkpoint VGG16. "
                "Use --checkpoint ou rode a etapa 1 primeiro."
            )
            sys.exit(1)
        log.info("Etapa 4 usara checkpoint: %s (val_acc via JSON se disponivel)", checkpoint_atual.name)
        novo = etapa_fine_tuning(checkpoint_atual, log)
        if novo:
            checkpoint_atual = novo

    # ── Etapa 5: avaliacao final ─────────────────────────────────────────────
    if 5 in args.etapas:
        if checkpoint_atual is None:
            checkpoint_atual = _checkpoint_melhor("vgg16")
        if checkpoint_atual is None:
            log.error(
                "Etapa 5 requer um checkpoint VGG16. "
                "Use --checkpoint ou rode as etapas 1-4 primeiro."
            )
            sys.exit(1)
        log.info("Etapa 5 usara checkpoint: %s", checkpoint_atual.name)
        etapa_avaliacao_final(checkpoint_atual, log)

    log.info("Pipeline concluido.")


if __name__ == "__main__":
    main()
