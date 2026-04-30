"""CLI do projeto Galaxy Classification.

Subcomandos:
    preprocessar  — Executa pipeline de pré-processamento (normalização, balanceamento, aumento)
    treinar       — Treina um modelo específico
    avaliar       — Avalia um modelo com pesos salvos
    explicar      — Gera visualização XAI para uma imagem
    benchmark     — Treina e avalia múltiplos modelos
    comparar      — Compara resultados de todos os modelos em docs/

Exemplos:
    python main.py treinar --modelo cnn --dataset decals --epocas 50
    python main.py avaliar --modelo resnet50 --pesos pesos/resnet50/resnet50_decals_ep50.pth
    python main.py benchmark --modelos cnn,resnet50,vit --dataset decals
    python main.py preprocessar --dataset sdss --tecnica smote
    python main.py explicar --modelo cnn --pesos pesos/cnn/cnn_decals_ep50.pth --imagem galaxy.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils.logger import configurar_logger_global
from utils.reproducibilidade import fixar_semente


# ---------------------------------------------------------------------------
# Subcomandos
# ---------------------------------------------------------------------------

def cmd_preprocessar(args: argparse.Namespace) -> None:
    """Pré-processa o dataset: normalização, balanceamento ou aumento de dados."""
    from dataset.carregador import CarregadorDataset
    from pre_processamento.aumento_de_dados import aplicar_aumento, salvar_dataset_h5
    from pre_processamento.balanceamento import executar_pipeline_balanceamento

    fixar_semente(args.seed)
    carregador = CarregadorDataset()
    imagens, rotulos = carregador.carregar(args.dataset)
    print(f"Dataset carregado: {imagens.shape[0]} amostras")

    if args.tecnica == "aumento":
        imagens, rotulos = aplicar_aumento(imagens, rotulos,
                                           fator_multiplicacao=args.fator, semente=args.seed)
        print(f"Após aumento: {imagens.shape[0]} amostras")
        saida = Path(args.saida) / f"{args.dataset}_aumento.h5"
    else:
        imagens, rotulos = executar_pipeline_balanceamento(
            imagens, rotulos, args.tecnica, semente=args.seed
        )
        print(f"Após balanceamento ({args.tecnica}): {imagens.shape[0]} amostras")
        saida = Path(args.saida) / f"{args.dataset}_{args.tecnica}.h5"

    salvar_dataset_h5(imagens, rotulos, saida, args.dataset)
    print(f"Salvo em: {saida}")


def cmd_treinar(args: argparse.Namespace) -> None:
    """Treina um modelo."""
    override = {}
    if args.dataset:
        override["dataset"] = args.dataset
    if args.epocas:
        override["epocas"] = args.epocas
    if args.batch_size:
        override["batch_size"] = args.batch_size
    if args.seed:
        override["seed"] = args.seed

    modelo = args.modelo.lower()
    print(f"Treinando modelo: {modelo}")

    if modelo == "cnn":
        from modelos.cnn.treino import treinar
    elif modelo == "resnet50":
        from modelos.resnet50.treino import treinar
    elif modelo == "efficientnet":
        from modelos.efficientnet.treino import treinar
    elif modelo == "vgg16":
        from modelos.vgg16.treino import treinar
    elif modelo == "vit":
        from modelos.vit.treino import treinar
    elif modelo == "dino":
        from modelos.dino.treino import treinar
    elif modelo == "multimodal":
        from modelos.multimodal.treino import treinar
    else:
        print(f"Modelo '{modelo}' não reconhecido. Use: cnn, resnet50, efficientnet, vgg16, vit, dino, multimodal")
        sys.exit(1)

    historico = treinar(config_override=override if override else None)
    print(f"\n{historico.resumo()}")


def cmd_avaliar(args: argparse.Namespace) -> None:
    """Avalia um modelo com pesos salvos."""
    from utils.metricas import formatar_para_markdown

    override = {}
    if args.dataset:
        override["dataset"] = args.dataset

    modelo = args.modelo.lower()
    caminho_pesos = Path(args.pesos)

    if not caminho_pesos.exists():
        print(f"Arquivo de pesos não encontrado: {caminho_pesos}")
        sys.exit(1)

    print(f"Avaliando modelo: {modelo} com pesos: {caminho_pesos}")

    if modelo == "cnn":
        from modelos.cnn.avaliacao import avaliar
    elif modelo == "resnet50":
        from modelos.resnet50.avaliacao import avaliar
    elif modelo == "efficientnet":
        from modelos.efficientnet.avaliacao import avaliar
    elif modelo == "vgg16":
        from modelos.vgg16.avaliacao import avaliar
    elif modelo == "vit":
        from modelos.vit.avaliacao import avaliar
    elif modelo == "dino":
        from modelos.dino.avaliacao import avaliar
    elif modelo == "multimodal":
        from modelos.multimodal.avaliacao import avaliar
    else:
        print(f"Modelo '{modelo}' não reconhecido.")
        sys.exit(1)

    resultado = avaliar(caminho_pesos, config_override=override if override else None)
    print(formatar_para_markdown(resultado))


def cmd_explicar(args: argparse.Namespace) -> None:
    """Gera visualização XAI para uma imagem."""
    import numpy as np
    from PIL import Image

    from modelos import obter_modelo
    from utils.visualizacao import plotar_sobreposicao_xai

    caminho_imagem = Path(args.imagem)
    caminho_pesos = Path(args.pesos)

    if not caminho_imagem.exists():
        print(f"Imagem não encontrada: {caminho_imagem}")
        sys.exit(1)

    imagem = np.array(Image.open(caminho_imagem).convert("RGB"))
    classificador = obter_modelo(args.modelo)

    import torch
    from pre_processamento.normalizacao import obter_transform_avaliacao
    from pre_processamento import config as pre_cfg

    transform = obter_transform_avaliacao()
    tensor = transform(imagem).unsqueeze(0)
    rede = classificador.construir(num_classes=10, tamanho_imagem=pre_cfg.TAMANHO_PADRAO)
    rede.load_state_dict(torch.load(caminho_pesos, map_location="cpu"))
    rede.eval()

    mapa = classificador.explicar(rede, tensor, classe_alvo=args.classe_alvo)

    nome_saida = args.saida or f"docs/{args.modelo}/xai/{caminho_imagem.stem}_xai.png"
    plotar_sobreposicao_xai(imagem, mapa, Path(nome_saida), titulo=f"XAI — {args.modelo}")
    print(f"XAI salvo em: {nome_saida}")


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Treina e avalia múltiplos modelos sequencialmente."""
    modelos = [m.strip() for m in args.modelos.split(",")]
    override = {}
    if args.dataset:
        override["dataset"] = args.dataset
    if args.epocas:
        override["epocas"] = args.epocas

    resultados = {}
    for modelo in modelos:
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {modelo.upper()}")
        print(f"{'='*60}")
        args.modelo = modelo
        cmd_treinar(args)

        # Tenta avaliar automaticamente se os pesos existirem
        dataset = override.get("dataset", "decals")
        epocas = override.get("epocas", 50)
        caminho_pesos = Path(f"pesos/{modelo}/{modelo}_{dataset}_ep{epocas}.pth")
        if caminho_pesos.exists():
            args.pesos = str(caminho_pesos)
            cmd_avaliar(args)

    # Comparativo
    args.resultados = "docs/"
    cmd_comparar(args)


def cmd_comparar(args: argparse.Namespace) -> None:
    """Lê resultados salvos e gera tabela comparativa."""
    import re
    from utils.metricas import ResultadoAvaliacao, CLASSES_GALAXY10
    from utils.visualizacao import plotar_comparativo_modelos
    import numpy as np

    dir_docs = Path(args.resultados if hasattr(args, "resultados") else "docs")
    resultados: dict[str, ResultadoAvaliacao] = {}

    for arq_resultado in sorted(dir_docs.glob("*/resultados.md")):
        nome_modelo = arq_resultado.parent.name
        conteudo = arq_resultado.read_text(encoding="utf-8")
        # Extrai acurácia do markdown gerado por formatar_para_markdown
        match_acc = re.search(r"Acurácia Top-1 \| ([\d.]+)", conteudo)
        match_f1 = re.search(r"F1 Macro \| ([\d.]+)", conteudo)
        if match_acc:
            from utils.metricas import ResultadoAvaliacao
            import numpy as np
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

    # Tabela comparativa
    dir_comp = dir_docs / "comparativo"
    dir_comp.mkdir(exist_ok=True)

    linhas = ["# Comparativo de Modelos\n\n",
              "| Modelo | Acurácia Top-1 | F1 Macro |\n",
              "|--------|---------------|----------|\n"]
    for nome, r in sorted(resultados.items(), key=lambda x: -x[1].acuracia):
        linhas.append(f"| {nome} | {r.acuracia:.4f} | {r.f1_macro:.4f} |\n")
    linhas.append(f"\n**Referência Astroformer:** ~0.9486 (DECaLS)\n")

    (dir_comp / "tabela_geral.md").write_text("".join(linhas), encoding="utf-8")
    print("".join(linhas))

    plotar_comparativo_modelos(resultados, dir_comp / "comparativo_acuracia.png")
    print(f"Comparativo salvo em: {dir_comp}")


# ---------------------------------------------------------------------------
# Parser principal
# ---------------------------------------------------------------------------

def _criar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="galaxy-classification",
        description="Classificação morfológica de galáxias — Galaxy10 SDSS/DECaLS",
    )
    parser.add_argument("--seed", type=int, default=42, help="Semente global")
    parser.add_argument("--dispositivo", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--verbose", action="store_true")

    sub = parser.add_subparsers(dest="comando", required=True)

    # preprocessar
    p_pre = sub.add_parser("preprocessar", help="Pré-processa o dataset")
    p_pre.add_argument("--dataset", required=True, choices=["sdss", "decals"])
    p_pre.add_argument("--tecnica", default="smote",
                       choices=["smote", "adasyn", "oversampling", "undersampling", "hibrido", "aumento"])
    p_pre.add_argument("--fator", type=int, default=2, help="Fator de multiplicação (só para aumento)")
    p_pre.add_argument("--saida", default="dataset/processados", help="Diretório de saída")
    p_pre.add_argument("--seed", type=int, default=42)

    # treinar
    p_train = sub.add_parser("treinar", help="Treina um modelo")
    p_train.add_argument("--modelo", required=True,
                         choices=["cnn", "resnet50", "efficientnet", "vgg16", "vit", "dino", "multimodal"])
    p_train.add_argument("--dataset", choices=["sdss", "decals"])
    p_train.add_argument("--epocas", type=int)
    p_train.add_argument("--batch-size", dest="batch_size", type=int)
    p_train.add_argument("--seed", type=int)

    # avaliar
    p_eval = sub.add_parser("avaliar", help="Avalia um modelo")
    p_eval.add_argument("--modelo", required=True,
                        choices=["cnn", "resnet50", "efficientnet", "vgg16", "vit", "dino", "multimodal"])
    p_eval.add_argument("--pesos", required=True, help="Caminho do arquivo .pth")
    p_eval.add_argument("--dataset", choices=["sdss", "decals"])

    # explicar
    p_xai = sub.add_parser("explicar", help="Gera visualização XAI")
    p_xai.add_argument("--modelo", required=True,
                       choices=["cnn", "resnet50", "efficientnet", "vgg16", "vit", "dino", "multimodal"])
    p_xai.add_argument("--pesos", required=True)
    p_xai.add_argument("--imagem", required=True, help="Caminho da imagem PNG/JPG")
    p_xai.add_argument("--classe-alvo", dest="classe_alvo", type=int, default=None)
    p_xai.add_argument("--saida", default=None, help="Caminho da imagem XAI de saída")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Treina e avalia múltiplos modelos")
    p_bench.add_argument("--modelos", required=True, help="Lista separada por vírgulas: cnn,resnet50,vit")
    p_bench.add_argument("--dataset", choices=["sdss", "decals"])
    p_bench.add_argument("--epocas", type=int)
    p_bench.add_argument("--seed", type=int)

    # comparar
    p_comp = sub.add_parser("comparar", help="Compara resultados salvos em docs/")
    p_comp.add_argument("--resultados", default="docs/", help="Diretório com subpastas de modelos")

    return parser


def main() -> None:
    parser = _criar_parser()
    args = parser.parse_args()

    import logging
    configurar_logger_global(logging.DEBUG if args.verbose else logging.INFO)

    fixar_semente(args.seed)

    cmds = {
        "preprocessar": cmd_preprocessar,
        "treinar": cmd_treinar,
        "avaliar": cmd_avaliar,
        "explicar": cmd_explicar,
        "benchmark": cmd_benchmark,
        "comparar": cmd_comparar,
    }
    cmds[args.comando](args)


if __name__ == "__main__":
    main()
