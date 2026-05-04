#!/usr/bin/python
"""CLI interativa do projeto Galaxy10 Classification.

Uso:
    python main.py              # Menu interativo
    python main.py --help       # Ajuda
"""

from __future__ import annotations

import sys
from pathlib import Path

from InquirerPy import inquirer
from InquirerPy.separator import Separator

from utils.config_loader import (
    carregar_config,
    listar_experimentos,
    listar_modelos,
    obter_config_modelo,
    obter_config_recursos,
    obter_experimento,
)
from utils.logger import configurar_logger_global
from utils.reproducibilidade import fixar_semente


# ---------------------------------------------------------------------------
# Mapeamento de modelos disponiveis
# ---------------------------------------------------------------------------

_NOMES_DISPLAY = {
    "cnn": "CNN Baseline",
    "resnet50": "ResNet50",
    "efficientnet": "EfficientNet-B0",
    "vgg16": "VGG16",
    "vit": "Vision Transformer (ViT)",
    "dino": "DINO (Self-Supervised)",
    "multimodal": "Multimodal (CNN + Tabular)",
}

_TECNICAS_PREPROCESS = [
    {"name": "Upscale (69px → 224px, LANCZOS)", "value": "upscale"},
    {"name": "SMOTE", "value": "smote"},
    {"name": "ADASYN", "value": "adasyn"},
    {"name": "Oversampling", "value": "oversampling"},
    {"name": "Undersampling", "value": "undersampling"},
    {"name": "Híbrido (SMOTE + undersampling)", "value": "hibrido"},
    {"name": "Aumento de dados (data augmentation)", "value": "aumento"},
]


# ---------------------------------------------------------------------------
# Ações do menu
# ---------------------------------------------------------------------------


def _acao_treinar(config: dict) -> None:
    """Fluxo interativo para treinar modelos."""
    modelos_disp = listar_modelos(config)

    # Escolher entre experimento pre-definido ou selecao manual
    exps = listar_experimentos(config)
    opcoes_fonte = [{"name": "Selecionar manualmente", "value": "manual"}]
    for exp in exps:
        opcoes_fonte.append({"name": f"Experimento: {exp}", "value": exp})

    fonte = inquirer.select(
        message="Como deseja configurar o treino?",
        choices=opcoes_fonte,
    ).execute()

    if fonte != "manual":
        _rodar_experimento(config, fonte)
        return

    # Selecao manual
    escolhas_modelos = [{"name": _NOMES_DISPLAY.get(m, m), "value": m} for m in modelos_disp]
    modelos = inquirer.checkbox(
        message="Selecione os modelos (espaco = marcar, enter = confirmar):",
        choices=escolhas_modelos,
        validate=lambda r: len(r) > 0,
        invalid_message="Selecione pelo menos um modelo.",
    ).execute()

    dataset = inquirer.select(
        message="Dataset de treino:",
        choices=["decals", "sdss"],
    ).execute()

    # Detectar versoes disponiveis
    from dataset.carregador import CarregadorDataset

    carregador = CarregadorDataset()
    versoes = carregador.listar_versoes_disponiveis(dataset)
    versao = inquirer.select(
        message="Versão do dataset:",
        choices=versoes,
    ).execute()

    cross = inquirer.confirm(
        message="Avaliar em dataset diferente (cross-dataset)?",
        default=False,
    ).execute()

    dataset_teste = None
    if cross:
        outro = "sdss" if dataset == "decals" else "decals"
        dataset_teste = inquirer.select(
            message="Dataset para avaliação:",
            choices=[outro],
            default=outro,
        ).execute()

    # Confirmar
    print(f"\n{'=' * 50}")
    print(f"  Modelos:  {', '.join(modelos)}")
    print(f"  Dataset:  {dataset} ({versao})")
    if dataset_teste:
        print(f"  Cross:    avaliar em {dataset_teste}")
    print(f"{'=' * 50}\n")

    confirmar = inquirer.confirm(message="Iniciar treino?", default=True).execute()
    if not confirmar:
        print("Cancelado.")
        return

    for modelo in modelos:
        print(f"\n{'=' * 60}")
        print(f"  TREINANDO: {_NOMES_DISPLAY.get(modelo, modelo).upper()}")
        print(f"{'=' * 60}")

        override = {
            "dataset": dataset,
            "versao_dataset": versao,
        }

        treinar_fn = _importar_treinar(modelo)
        historico = treinar_fn(config_override=override)
        print(f"  {historico.resumo()}")

        if dataset_teste:
            print(f"\n  Avaliando cross-dataset em {dataset_teste}...")
            avaliar_fn = _importar_avaliar(modelo)
            # Encontrar o .pth mais recente
            pesos = _encontrar_pesos_recentes(modelo)
            if pesos:
                avaliar_fn(pesos, config_override={"dataset_teste": dataset_teste})


def _acao_avaliar(config: dict) -> None:
    """Fluxo interativo para avaliar um modelo."""
    modelos_disp = listar_modelos(config)

    modelo = inquirer.select(
        message="Modelo a avaliar:",
        choices=[{"name": _NOMES_DISPLAY.get(m, m), "value": m} for m in modelos_disp],
    ).execute()

    # Listar pesos disponiveis
    dir_pesos = Path("pesos") / modelo
    if not dir_pesos.exists() or not list(dir_pesos.glob("*.pth")):
        print(f"Nenhum peso encontrado em {dir_pesos}/")
        return

    arquivos_pth = sorted(dir_pesos.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    escolhas_pesos = [{"name": p.name, "value": p} for p in arquivos_pth]
    caminho_pesos = inquirer.select(
        message="Arquivo de pesos:",
        choices=escolhas_pesos,
    ).execute()

    cross = inquirer.confirm(
        message="Avaliar em dataset diferente (cross-dataset)?",
        default=False,
    ).execute()

    override = {}
    if cross:
        dataset_teste = inquirer.select(
            message="Dataset para avaliação:",
            choices=["decals", "sdss"],
        ).execute()
        override["dataset_teste"] = dataset_teste

    avaliar_fn = _importar_avaliar(modelo)
    resultado = avaliar_fn(caminho_pesos, config_override=override if override else None)

    from utils.metricas import formatar_para_markdown

    print(formatar_para_markdown(resultado))


def _acao_preprocessar(config: dict) -> None:
    """Fluxo interativo para pre-processar dataset."""
    dataset = inquirer.select(
        message="Dataset:",
        choices=["decals", "sdss"],
    ).execute()

    tecnica = inquirer.select(
        message="Técnica de pré-processamento:",
        choices=_TECNICAS_PREPROCESS,
    ).execute()

    fator = 2
    if tecnica == "aumento":
        fator = int(
            inquirer.text(
                message="Fator de multiplicação:",
                default="2",
                validate=lambda v: v.isdigit() and int(v) > 0,
            ).execute()
        )

    seed = config.get("global", {}).get("seed", 42)

    print(f"\nPré-processando {dataset} com {tecnica}...")

    from dataset.carregador import CarregadorDataset
    from pre_processamento.aumento_de_dados import aplicar_aumento, salvar_dataset_h5
    from pre_processamento.balanceamento import executar_pipeline_balanceamento
    from pre_processamento.upscale import aplicar_upscale

    fixar_semente(seed)
    carregador = CarregadorDataset()
    imagens, rotulos = carregador.carregar(dataset)
    print(f"Dataset carregado: {imagens.shape[0]} amostras, shape {imagens.shape[1:]}")

    if tecnica == "upscale":
        tamanho = config.get("global", {}).get("tamanho_imagem", 224)
        print(f"Upscale {imagens.shape[1]}px → {tamanho}px (LANCZOS)...")
        imagens = aplicar_upscale(imagens, tamanho_alvo=tamanho)
        print(f"Após upscale: shape {imagens.shape[1:]}")
        saida = Path("dataset/processados") / f"{dataset}_upscale.h5"
    elif tecnica == "aumento":
        imagens, rotulos = aplicar_aumento(
            imagens, rotulos, fator_multiplicacao=fator, semente=seed
        )
        print(f"Após aumento: {imagens.shape[0]} amostras")
        saida = Path("dataset/processados") / f"{dataset}_aumento.h5"
    else:
        imagens, rotulos = executar_pipeline_balanceamento(imagens, rotulos, tecnica, semente=seed)
        print(f"Após balanceamento ({tecnica}): {imagens.shape[0]} amostras")
        saida = Path("dataset/processados") / f"{dataset}_{tecnica}.h5"

    salvar_dataset_h5(imagens, rotulos, saida, dataset)
    print(f"Salvo em: {saida}")


def _acao_amostras_xai(config: dict) -> None:
    """Extrai amostras do dataset por classe e salva arquivos individuais por categoria."""
    import warnings

    from dataset.carregador import CarregadorDataset, resolver_nome_dataset
    from utils.amostragem_dataset import extrair_amostras_por_classe, salvar_imagens_por_classe

    modelos_disp = listar_modelos(config)
    modelo = inquirer.select(
        message="Modelo (define subpasta de saída):",
        choices=[{"name": _NOMES_DISPLAY.get(m, m), "value": m} for m in modelos_disp],
    ).execute()

    dataset = inquirer.select(
        message="Dataset:",
        choices=["decals", "sdss"],
    ).execute()

    carregador = CarregadorDataset()
    versoes = carregador.listar_versoes_disponiveis(dataset)
    versao = inquirer.select(
        message="Versão do dataset:",
        choices=versoes,
    ).execute()

    cfg_xai = config.get("xai", {})
    n_por_classe = cfg_xai.get("amostras_por_classe", 50)
    dir_saida = cfg_xai.get("diretorio_saida", "docs/xai")
    seed = config.get("global", {}).get("seed", 42)

    nome_ds = resolver_nome_dataset(dataset, versao)
    print(f"\nCarregando {nome_ds}...")
    imagens, rotulos = carregador.carregar(nome_ds)
    print(f"Dataset: {imagens.shape[0]} amostras, shape {imagens.shape[1:]}")

    print(f"Extraindo até {n_por_classe} amostras por classe...")
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        amostras = extrair_amostras_por_classe(imagens, rotulos, n_por_classe, semente=seed)

    for av in avisos:
        print(f"  AVISO: {av.message}")

    dir_modelo = Path(dir_saida) / modelo
    caminhos = salvar_imagens_por_classe(amostras, dir_modelo)
    total = sum(len(v) for v in caminhos.values())
    print(f"{total} imagens salvas em: {dir_modelo}/")
    for c, paths in caminhos.items():
        print(f"  classe_{c:02d}: {len(paths)} imagens")


def _acao_xai_em_lote(config: dict) -> None:
    """Roda XAI em todas as amostras extraídas e salva grade em docs/{modelo}/xai.png."""
    import numpy as np
    from PIL import Image

    from modelos import obter_modelo
    from pre_processamento.normalizacao import obter_transform_avaliacao
    from utils.checkpoint import carregar_checkpoint
    from utils.visualizacao import plotar_grade_xai_por_classe

    cfg_xai = config.get("xai", {})
    dir_xai = Path(cfg_xai.get("diretorio_saida", "docs/xai"))

    modelos_disp = listar_modelos(config)
    modelo = inquirer.select(
        message="Modelo:",
        choices=[{"name": _NOMES_DISPLAY.get(m, m), "value": m} for m in modelos_disp],
    ).execute()

    dir_amostras = dir_xai / modelo
    if not dir_amostras.exists() or not any(p for p in dir_amostras.iterdir() if p.is_dir()):
        print(f"Nenhuma amostra encontrada em {dir_amostras}/")
        print("Execute primeiro 'Extrair amostras do dataset (XAI)'.")
        return

    dir_pesos = Path("pesos") / modelo
    if not dir_pesos.exists() or not list(dir_pesos.glob("*.pth")):
        print(f"Nenhum peso encontrado em {dir_pesos}/")
        return

    arquivos_pth = sorted(dir_pesos.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    caminho_pesos = inquirer.select(
        message="Arquivo de pesos:",
        choices=[{"name": p.name, "value": p} for p in arquivos_pth],
    ).execute()

    params = obter_config_modelo(modelo, config)
    tamanho = params.get("tamanho_imagem", 224)

    classificador = obter_modelo(modelo)

    import torch  # noqa: F401 — garante que está disponível

    rede = classificador.construir(num_classes=10, tamanho_imagem=tamanho)
    carregar_checkpoint(caminho_pesos, rede)
    rede.eval()

    transform = obter_transform_avaliacao(tamanho_imagem=tamanho)

    pastas_classes = sorted(p for p in dir_amostras.iterdir() if p.is_dir())
    total = sum(len(list(p.glob("*.png"))) for p in pastas_classes)
    print(f"\nProcessando {total} imagens em {len(pastas_classes)} classes...")

    resultados: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    processados = 0

    for pasta in pastas_classes:
        nome_classe = pasta.name
        resultados[nome_classe] = []
        for arq in sorted(pasta.glob("*.png")):
            with Image.open(arq) as pil_img:
                img_np = np.array(pil_img.convert("RGB"))
            tensor = transform(img_np).unsqueeze(0)
            mapa = classificador.explicar(rede, tensor, classe_alvo=None)
            resultados[nome_classe].append((img_np, mapa))
            processados += 1
            print(f"\r  {processados}/{total}", end="", flush=True)

    print()
    imagens_por_linha = cfg_xai.get("imagens_por_linha", 5)
    dir_saida_xai = Path("docs") / modelo / "xai"
    plotar_grade_xai_por_classe(resultados, dir_saida_xai, imagens_por_linha=imagens_por_linha)
    print(f"Imagens XAI salvas em: {dir_saida_xai}/ (uma por classe)")


def _acao_explicar(config: dict) -> None:
    """Fluxo interativo para gerar XAI."""
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    from modelos import obter_modelo
    from pre_processamento.normalizacao import obter_transform_avaliacao
    from utils.checkpoint import carregar_checkpoint
    from utils.visualizacao import plotar_sobreposicao_xai

    modelos_disp = listar_modelos(config)
    modelo = inquirer.select(
        message="Modelo:",
        choices=[{"name": _NOMES_DISPLAY.get(m, m), "value": m} for m in modelos_disp],
    ).execute()

    dir_pesos = Path("pesos") / modelo
    if not dir_pesos.exists() or not list(dir_pesos.glob("*.pth")):
        print(f"Nenhum peso encontrado em {dir_pesos}/")
        return

    arquivos_pth = sorted(dir_pesos.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    caminho_pesos = inquirer.select(
        message="Arquivo de pesos:",
        choices=[{"name": p.name, "value": p} for p in arquivos_pth],
    ).execute()

    caminho_imagem = inquirer.filepath(
        message="Caminho da imagem (PNG/JPG):",
        validate=lambda p: Path(p).is_file(),
        invalid_message="Arquivo não encontrado ou caminho inválido.",
    ).execute()

    try:
        with Image.open(caminho_imagem) as img:
            imagem = np.array(img.convert("RGB"))
    except UnidentifiedImageError:
        print(f"Arquivo inválido: '{caminho_imagem}' não é uma imagem PNG/JPG válida.")
        return
    except OSError as erro:
        print(f"Falha ao abrir imagem '{caminho_imagem}': {erro}")
        return

    classificador = obter_modelo(modelo)

    import torch

    params = obter_config_modelo(modelo, config)
    tamanho = params.get("tamanho_imagem", 224)

    transform = obter_transform_avaliacao(tamanho_imagem=tamanho)
    tensor = transform(imagem).unsqueeze(0)
    rede = classificador.construir(num_classes=10, tamanho_imagem=tamanho)
    carregar_checkpoint(caminho_pesos, rede)
    rede.eval()

    mapa = classificador.explicar(rede, tensor, classe_alvo=None)
    nome_saida = Path(f"docs/{modelo}/xai/{Path(caminho_imagem).stem}_xai.png")
    plotar_sobreposicao_xai(imagem, mapa, nome_saida, titulo=f"XAI — {modelo}")
    print(f"XAI salvo em: {nome_saida}")


def _acao_info(config: dict) -> None:
    """Mostra informações do sistema e datasets."""
    from utils.recursos import info_recursos
    from dataset.carregador import CarregadorDataset

    info = info_recursos()
    cfg_rec = obter_config_recursos(config)

    print(f"\n{'=' * 50}")
    print("  INFORMAÇÕES DO SISTEMA")
    print(f"{'=' * 50}")
    print(f"  CPUs:           {info['cpu_count']}")
    print(f"  RAM total:      {info.get('ram_total_gb', 'N/A')} GB")
    print(f"  RAM disponível: {info.get('ram_disponivel_gb', 'N/A')} GB")
    print(f"  CUDA:           {'Sim' if info['cuda_disponivel'] else 'Não'}")

    for gpu in info.get("gpus", []):
        print(f"  GPU {gpu['indice']}:         {gpu['nome']}")
        print(f"    VRAM total:   {gpu['vram_total_gb']} GB")
        print(f"    VRAM livre:   {gpu['vram_livre_gb']} GB")

    print(f"\n  CONFIG DE RECURSOS:")
    print(f"    Dispositivo:       {cfg_rec.get('dispositivo', 'auto')}")
    print(f"    Max GPU memória:   {cfg_rec.get('max_gpu_memoria_gb', 'sem limite')}")
    print(f"    Mixed precision:   {cfg_rec.get('mixed_precision', True)}")
    print(f"    Num workers:       {cfg_rec.get('num_workers', 'auto')}")

    print(f"\n  DATASETS DISPONÍVEIS:")
    carregador = CarregadorDataset()
    for ds in ["sdss", "decals"]:
        versoes = carregador.listar_versoes_disponiveis(ds)
        print(f"    {ds}: {', '.join(versoes)}")

    print(f"\n  EXPERIMENTOS PRÉ-DEFINIDOS:")
    for exp in listar_experimentos(config):
        dados = obter_experimento(exp, config)
        modelos = dados.get("modelos", [])
        datasets = dados.get("datasets", [dados.get("treinar_em", "?")])
        versao = dados.get("versao_dataset", "raw")
        print(f"    {exp}: {', '.join(modelos)} | {', '.join(datasets)} ({versao})")

    print()


def _acao_historico(config: dict) -> None:
    """Mostra histórico de runs."""
    from utils.experimento import carregar_historico_todos

    todos = carregar_historico_todos(Path("pesos"))
    if not todos:
        print("Nenhum histórico de runs encontrado.")
        return

    print(f"\n{'=' * 80}")
    print("  HISTÓRICO DE RUNS")
    print(f"{'=' * 80}")

    for modelo, runs in todos.items():
        print(f"\n  {modelo.upper()} ({len(runs)} runs):")
        print(f"  {'─' * 76}")
        for run in runs:
            nome = run.get("nome_experimento", "?")
            ts = run.get("timestamp", "?")[:19]
            acc = run.get("melhor_val_acc")
            epocas = run.get("epocas_totais", "?")
            parou = run.get("parou_cedo", False)
            acc_str = f"{acc:.4f}" if acc is not None else "N/A"
            es_str = " (early stop)" if parou else ""
            print(f"    {ts} | acc={acc_str} | ep={epocas}{es_str} | {nome}")

    print()


def _acao_comparar(config: dict) -> None:
    """Compara resultados salvos em docs/."""
    import re
    import numpy as np
    from utils.metricas import ResultadoAvaliacao
    from utils.visualizacao import plotar_comparativo_modelos

    dir_docs = Path("docs")
    resultados: dict[str, ResultadoAvaliacao] = {}

    for arq in sorted(dir_docs.glob("*/resultados.md")):
        nome_modelo = arq.parent.name
        conteudo = arq.read_text(encoding="utf-8")
        match_acc = re.search(r"Acurácia Top-1 \| ([\d.]+)", conteudo)
        match_f1 = re.search(r"F1 Macro \| ([\d.]+)", conteudo)
        if match_acc:
            resultado = ResultadoAvaliacao(
                acuracia=float(match_acc.group(1)),
                acuracia_top5=None,
                precisao_macro=0.0,
                recall_macro=0.0,
                f1_macro=float(match_f1.group(1)) if match_f1 else 0.0,
                matriz_confusao=np.zeros((10, 10)),
                acuracia_por_classe={},
                relatorio="",
                nome_modelo=nome_modelo,
            )
            resultados[nome_modelo] = resultado

    if not resultados:
        print("Nenhum resultado encontrado em docs/*/resultados.md")
        return

    dir_comp = dir_docs / "comparativo"
    dir_comp.mkdir(exist_ok=True)

    linhas = [
        "# Comparativo de Modelos\n\n",
        "| Modelo | Acurácia Top-1 | F1 Macro |\n",
        "|--------|---------------|----------|\n",
    ]
    for nome, r in sorted(resultados.items(), key=lambda x: -x[1].acuracia):
        linhas.append(f"| {nome} | {r.acuracia:.4f} | {r.f1_macro:.4f} |\n")
    linhas.append(f"\n**Referência Astroformer:** ~0.9486 (DECaLS)\n")

    (dir_comp / "tabela_geral.md").write_text("".join(linhas), encoding="utf-8")
    print("".join(linhas))

    plotar_comparativo_modelos(resultados, dir_comp / "comparativo_acuracia.png")
    print(f"Comparativo salvo em: {dir_comp}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _importar_treinar(modelo: str):
    """Importa dinamicamente a funcao treinar() de um modelo."""
    modulo = __import__(f"modelos.{modelo}.treino", fromlist=["treinar"])
    return modulo.treinar


def _importar_avaliar(modelo: str):
    """Importa dinamicamente a funcao avaliar() de um modelo."""
    modulo = __import__(f"modelos.{modelo}.avaliacao", fromlist=["avaliar"])
    return modulo.avaliar


def _encontrar_pesos_recentes(modelo: str) -> Path | None:
    """Retorna o .pth mais recente de um modelo."""
    dir_pesos = Path("pesos") / modelo
    if not dir_pesos.exists():
        return None
    arquivos = sorted(dir_pesos.glob("*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
    return arquivos[0] if arquivos else None


# Chaves que controlam estrutura do experimento — não são overrides de hiperparâmetros
_CHAVES_ESTRUTURA_EXP = frozenset(
    {"modelos", "datasets", "versao_dataset", "treinar_em", "avaliar_em", "modo_treino"}
)


def _rodar_experimento(config: dict, nome_exp: str) -> None:
    """Roda um experimento pre-definido do config.yaml."""
    exp = obter_experimento(nome_exp, config)
    modelos = exp.get("modelos", [])
    datasets = exp.get("datasets", [])
    treinar_em = exp.get("treinar_em")
    avaliar_em = exp.get("avaliar_em")
    versao = exp.get("versao_dataset", "raw")
    modo_treino = exp.get("modo_treino")

    if treinar_em:
        datasets = [treinar_em]

    # Hiperparâmetros extras do experimento (ex: epocas_congelado: 3 para sweep)
    overrides_exp = {k: v for k, v in exp.items() if k not in _CHAVES_ESTRUTURA_EXP}

    print(f"\n{'=' * 60}")
    print(f"  EXPERIMENTO: {nome_exp}")
    print(f"  Modelos: {', '.join(modelos)}")
    print(f"  Datasets: {', '.join(datasets)} ({versao})")
    if avaliar_em:
        print(f"  Cross-dataset: avaliar em {avaliar_em}")
    if modo_treino:
        print(f"  Modo treino: {modo_treino}")
    if overrides_exp:
        print(f"  Overrides: {overrides_exp}")
    print(f"{'=' * 60}\n")

    confirmar = inquirer.confirm(message="Iniciar?", default=True).execute()
    if not confirmar:
        return

    for modelo in modelos:
        for dataset in datasets:
            print(f"\n--- {modelo.upper()} em {dataset} ({versao}) ---")
            override = {"dataset": dataset, "versao_dataset": versao, **overrides_exp}

            if modo_treino == "distilacao":
                from modelos.vgg16.distilacao import treinar_com_distilacao
                historico = treinar_com_distilacao(config_override=override)
            else:
                treinar_fn = _importar_treinar(modelo)
                historico = treinar_fn(config_override=override)

            print(f"  {historico.resumo()}")

            if avaliar_em:
                print(f"  Avaliando cross-dataset em {avaliar_em}...")
                pesos = _encontrar_pesos_recentes(modelo)
                if pesos:
                    avaliar_fn = _importar_avaliar(modelo)
                    avaliar_fn(pesos, config_override={"dataset_teste": avaliar_em})


# ---------------------------------------------------------------------------
# Menu principal
# ---------------------------------------------------------------------------


def main() -> None:
    import logging

    configurar_logger_global(logging.INFO)

    config = carregar_config()
    fixar_semente(config.get("global", {}).get("seed", 42))

    print()
    print("=" * 50)
    print("  Galaxy10 Classification")
    print("=" * 50)
    print()

    while True:
        acao = inquirer.select(
            message="O que deseja fazer?",
            choices=[
                {"name": "Treinar modelos", "value": "treinar"},
                {"name": "Avaliar modelo", "value": "avaliar"},
                {"name": "Pré-processar dataset", "value": "preprocessar"},
                {"name": "Gerar XAI (explicabilidade)", "value": "explicar"},
                {"name": "Extrair amostras do dataset (XAI)", "value": "amostras_xai"},
                {"name": "XAI em lote (amostras extraídas)", "value": "xai_em_lote"},
                {"name": "Comparar resultados", "value": "comparar"},
                {"name": "Ver histórico de runs", "value": "historico"},
                {"name": "Ver informações do sistema", "value": "info"},
                Separator(),
                {"name": "Sair", "value": "sair"},
            ],
        ).execute()

        if acao == "sair":
            break

        acoes = {
            "treinar": _acao_treinar,
            "avaliar": _acao_avaliar,
            "preprocessar": _acao_preprocessar,
            "explicar": _acao_explicar,
            "amostras_xai": _acao_amostras_xai,
            "xai_em_lote": _acao_xai_em_lote,
            "comparar": _acao_comparar,
            "historico": _acao_historico,
            "info": _acao_info,
        }
        acoes[acao](config)


if __name__ == "__main__":
    main()
