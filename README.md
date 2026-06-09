# A Look Outside — README de Entrega de Artefatos

Este documento descreve, de forma objetiva, os artefatos entregues no projeto, sua função e os parâmetros relevantes para execução/reprodução.

## 1) Link da pasta de entrega

> **Preencher antes da submissão final:**  
> **LINK:** `COLE_AQUI_O_LINK_PUBLICO_DA_PASTA`  
> (GitHub, Google Drive ou Dropbox com permissão de leitura)

## 2) Estrutura dos artefatos submetidos

```text
.
├── README.md
├── config.yaml
├── main.py
├── benchmark/
│   ├── main.py
│   ├── orchestrator.py
│   └── recommendations.py
├── dataset/
│   ├── main.py
│   ├── input_output.py
│   └── balancing/
├── machine-learning/
│   ├── main.py
│   ├── pipeline.py
│   ├── runs/                         # logs, checkpoints e métricas por execução
│   └── models/
├── xai/
│   ├── main.py
│   └── methods/
└── docs/                             # resultados consolidados (métricas, figuras e tabelas)
```

## 3) Inventário objetivo dos artefatos

| Artefato | Função no projeto | Parâmetros de execução (quando aplicável) |
|---|---|---|
| `main.py` | Orquestrador interativo para iniciar benchmark, dataset, machine learning e XAI. | Sem parâmetros de linha de comando; usa `config.yaml`. |
| `benchmark/main.py` | Executa benchmarks declarativos fim a fim (dataset → treino → avaliação → XAI/recomendações). | `--benchmark` (`-b`): nome do benchmark em `config.yaml`; `--yes` (`-y`): pula confirmação interativa. |
| `dataset/main.py` | Balanceamento e análise de datasets `.h5`. | Interativo; usa `config.yaml` (ex.: `paths.raw_dataset_directory`, `paths.processed_dataset_directory`, `random_seed`, `dataset_analysis.output_directory`). |
| `machine-learning/main.py` | Treino e avaliação dos modelos. Gera runs reproduzíveis. | Interativo; usa `config.yaml` (ex.: `training.*`, `models.*`, `pipelines.*`) e `machine-learning/my-computer.yaml` para limites de hardware. |
| `xai/main.py` | Geração de explicações visuais a partir de runs treinadas. | Interativo; usa `config.yaml` (ex.: `xai.default_methods`, `paths.xai_output_directory`). |
| `config.yaml` | Configuração central de caminhos, treino, modelos, pipelines, benchmarks e XAI. | Arquivo-base para todas as execuções. |
| `docs/runs.jsonl` | Registro consolidado das execuções (manifesto de runs). | Gerado pelas execuções de ML/benchmark. |
| `docs/leaderboard.md` e `docs/leaderboard.csv` | Ranking consolidado dos resultados experimentais. | Gerados a partir de `runs.jsonl`. |
| `docs/models/<modelo>/<run>/metrics.md` | Métricas por execução. | Saída gerada automaticamente pela etapa de documentação. |
| `docs/models/<modelo>/<run>/classification_report.md` | Relatório por classe. | Saída gerada automaticamente. |
| `docs/models/<modelo>/<run>/*.png` | Figuras auxiliares (matriz de confusão, curvas ROC/PR, learning curves etc.). | Saída gerada automaticamente. |
| `docs/dataset/<dataset>/` | Relatórios de análise dos datasets (estatísticas, distribuição, histogramas, mosaicos). | Saída gerada por `dataset/main.py` e benchmarks. |
| `machine-learning/runs/<run>/run.log` | Log detalhado da execução de treino/avaliação. | Gerado automaticamente por run. |
| `machine-learning/runs/<run>/metrics.json` | Métricas em formato estruturado para auditoria/reprodução. | Gerado automaticamente por run. |
| `machine-learning/runs/<run>/config.yaml` | Snapshot da configuração efetiva usada na run. | Gerado automaticamente por run. |
| `machine-learning/runs/<run>/*.pth` | Pesos/modelo treinado. | Gerado para modelos treináveis. |

## 4) Localização explícita dos resultados relevantes

- **Dados de entrada (`.h5`)**: `dataset/raw/`
- **Dados processados/balanceados (`.h5`)**: `dataset/processed/`
- **Modelos treinados (pesos)**: `machine-learning/runs/<run>/*.pth`
- **Logs de execução**: `machine-learning/runs/<run>/run.log`
- **Métricas estruturadas**: `machine-learning/runs/<run>/metrics.json`
- **Métricas em relatório**: `docs/models/<modelo>/<run>/metrics.md`
- **Figuras auxiliares**: `docs/models/<modelo>/<run>/*.png` e `docs/dataset/<dataset>/*.png`
- **Tabelas auxiliares**: `docs/leaderboard.csv`, `docs/models/<modelo>/<run>/classification_report.md`, `docs/dataset/<dataset>/class_statistics.csv`

## 5) Execução e reprodução

### 5.1 Preparação do ambiente

```bash
mise install
uv sync
```

### 5.2 Execuções principais

```bash
uv run python main.py
uv run python dataset/main.py
uv run python machine-learning/main.py
uv run python xai/main.py
```

### 5.3 Benchmark declarativo (não interativo)

```bash
uv run python benchmark/main.py --benchmark everything --yes
```

## 6) Observações para avaliação

- A pasta de entrega deve permanecer **acessível em leitura** até o fim da avaliação.
- Este README referencia explicitamente os diretórios onde estão **dados, modelos, códigos, logs, métricas, figuras e tabelas**.
- Os artefatos submetidos devem permitir rastrear e comprovar os resultados apresentados no artigo.
