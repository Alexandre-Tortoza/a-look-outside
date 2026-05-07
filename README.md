# A Look Outside

Projeto para classificacao morfologica de galaxias usando datasets em formato H5, tecnicas de balanceamento, pipelines de machine learning e geracao de artefatos de explicabilidade.

## Visao Geral

O projeto sera organizado em quatro areas principais:

- `dataset/`: leitura dos datasets originais e geracao de versoes balanceadas.
- `machine-learning/`: treinamento, avaliacao e reproducao de modelos.
- `xai/`: extracao de amostras e geracao de explicacoes visuais.
- `docs/`: relatorios, metricas, graficos e imagens finais das runs.

Todo o codigo do projeto deve ser escrito em ingles, incluindo nomes de arquivos, funcoes, classes, variaveis, comentarios, docstrings, mensagens de erro, logs e chaves de configuracao. O README pode permanecer em portugues.

## Estrutura Planejada

```text
.
├── README.md
├── main.py
├── config.yaml
├── mise.toml
├── pyproject.toml
├── uv.lock
├── dataset/
│   ├── main.py
│   ├── input_output.py
│   ├── raw/
│   │   ├── sdss.h5
│   │   └── decals.h5
│   ├── processed/
│   └── balancing/
│       ├── registry.py
│       ├── smote.py
│       ├── random_over_sampling.py
│       └── random_under_sampling.py
├── machine-learning/
│   ├── main.py
│   ├── my-computer.yaml
│   ├── data_loading.py
│   ├── pipeline.py
│   ├── run_storage.py
│   ├── documentation_storage.py
│   ├── runs/
│   └── models/
│       ├── registry.py
│       ├── dino.py
│       ├── vgg16.py
│       ├── efficientnet.py
│       ├── resnet50.py
│       └── k_nearest_neighbors.py
├── xai/
│   ├── main.py
│   ├── sample_extraction.py
│   ├── explanation_generation.py
│   ├── artifact_storage.py
│   └── methods/
│       ├── registry.py
│       └── gradient_class_activation_mapping.py
└── docs/
```

## Datasets

Os datasets originais devem ficar em:

```text
dataset/raw/sdss.h5
dataset/raw/decals.h5
```

Cada arquivo H5 deve conter as chaves:

- `images`: imagens do dataset.
- `ans`: rotulos inteiros das classes.

Os datasets processados serao salvos em `dataset/processed/`, mantendo a mesma convencao de chaves.

## Ambiente

O projeto deve usar `mise-en-place` e `uv` para garantir compatibilidade de ambiente e instalacao reproduzivel.

- `mise.toml`: define a versao do Python e comandos padronizados do projeto.
- `pyproject.toml`: define dependencias, metadados e ferramentas Python.
- `uv.lock`: trava as versoes resolvidas das dependencias.

Fluxo esperado:

```bash
mise install
uv sync
uv run python main.py
```

As CLIs internas tambem devem ser executaveis com `uv run`:

```bash
uv run python dataset/main.py
uv run python machine-learning/main.py
uv run python xai/main.py
```

## Orquestrador Principal

O arquivo `main.py`, na raiz do projeto, deve funcionar como orquestrador interativo.

Ele deve permitir navegar para os modulos principais:

- dataset balancing;
- machine learning;
- XAI.

O orquestrador deve ser desacoplado. Ele apenas aponta ou delega para os modulos, sem concentrar a logica deles. Cada CLI interna deve continuar funcionando diretamente:

```bash
uv run python dataset/main.py
uv run python machine-learning/main.py
uv run python xai/main.py
```

## Balanceamento de Datasets

A CLI de balanceamento ficara em `dataset/main.py` e deve ser interativa. O usuario deve selecionar os datasets e os metodos usando checkboxes ou listas de selecao, sem precisar escrever todos os parametros no comando.

Exemplo:

```bash
python dataset/main.py
```

Metodos iniciais:

- `smote`: SMOTE para imagens, achatando cada imagem, interpolando vizinhos da mesma classe e restaurando o formato original.
- `random_over_sampling`: duplicacao aleatoria das classes minoritarias.
- `random_under_sampling`: selecao aleatoria das classes majoritarias.

Exemplos de saida:

```text
dataset/processed/sdss_smote.h5
dataset/processed/decals_random_over_sampling.h5
dataset/processed/sdss_smote_random_under_sampling.h5
```

## Machine Learning

A CLI de machine learning ficara em `machine-learning/main.py`. Ela deve ser interativa e executar pipelines configuraveis, permitindo selecionar modelos, datasets raw ou processed e pipelines usando checkboxes ou listas de selecao.

Comando principal:

```bash
python machine-learning/main.py
```

Modelos iniciais:

- `dino`
- `vgg16`
- `efficientnet`
- `resnet50`
- `k_nearest_neighbors`

Cada modelo deve ficar em um arquivo separado dentro de `machine-learning/models/`, com registro central em `machine-learning/models/registry.py`.

## Configuracao do Computador

Na primeira execucao da CLI de machine learning, o projeto deve gerar automaticamente:

```text
machine-learning/my-computer.yaml
```

Esse arquivo deve conter as especificacoes detectadas do computador e os limites de recursos que a pessoa quer dedicar aos treinos.

Por padrao, a configuracao deve alocar todos os recursos disponiveis, incluindo GPU quando houver suporte. Se a pessoa nao quiser usar todos os recursos, ela pode editar `machine-learning/my-computer.yaml`.

Conteudo esperado:

```yaml
computer:
  processor_name: auto_detected
  physical_core_count: auto_detected
  logical_core_count: auto_detected
  total_memory_gigabytes: auto_detected
  gpu_available: auto_detected
  gpu_name: auto_detected
  gpu_memory_gigabytes: auto_detected

resource_limits:
  use_gpu: true
  gpu_device: auto
  maximum_gpu_memory_gigabytes: null
  maximum_cpu_worker_count: null
  maximum_memory_gigabytes: null
  use_mixed_precision: true
```

Valores `null` em limites significam "usar o maximo disponivel".

## Runs

Cada execucao modelo x dataset deve criar uma pasta em:

```text
machine-learning/runs/<model-name>-<dataset-name>-<day-month-year>/
```

Exemplo:

```text
machine-learning/runs/vgg16-sdss-smote-26-10-2001/
```

Dentro da pasta da run devem existir:

- `run.log`: logs completos da execucao.
- `config.yaml`: copia da configuracao efetiva, no mesmo formato da configuracao de entrada.
- `<model-name>-<dataset-name>-<run-date>.pth`: checkpoint para reproducao.
- `metrics.json`: metricas da run.
- artefatos brutos necessarios para reproducibilidade.

Se uma pasta de run com o mesmo nome ja existir, a implementacao deve adicionar um sufixo incremental, por exemplo:

```text
vgg16-sdss-smote-26-10-2001-2/
```

## Documentacao das Runs

Os relatorios finais das runs devem ser salvos em `docs/`:

```text
docs/<model-name>/<run-name>-<dataset-name>/
```

Exemplo:

```text
docs/k_nearest_neighbors/26-10-2001-sdss-smote/
```

Arquivos esperados:

- `summary.md`: resumo da run, dataset, modelo, configuracoes principais, metricas e indice de artefatos.
- `metrics.md`: tabela de metricas.
- `classification_report.md`: relatorio por classe.
- `confusion_matrix.png`: matriz de confusao.
- `learning_curves.png`: curvas de aprendizado para modelos treinaveis.
- `class_distribution.png`: distribuicao de classes.

Para `k_nearest_neighbors`, podem ser adicionados:

- `neighbor_examples.png`
- `distance_distribution.png`

## XAI

A CLI de explicabilidade ficara em `xai/main.py`. Ela deve ser interativa e permitir extrair amostras e gerar explicacoes em lote usando selecoes por checkbox ou lista.

Comando:

```bash
python xai/main.py
```

Estrutura de saida:

```text
xai/<model-name>/<dataset-name>/samples/*.png
xai/<model-name>/<dataset-name>/xai/*.png
```

Exemplo:

```text
xai/k_nearest_neighbors/sdss_smote/samples/spiral_barred_0001.png
xai/k_nearest_neighbors/sdss_smote/xai/spiral_barred_0001.png
```

Quando os nomes das classes estiverem definidos em `config.yaml`, eles devem ser usados nos nomes dos arquivos. Caso contrario, deve ser usado o formato:

```text
class_0_0001.png
```

Suporte inicial planejado:

- Grad-CAM para modelos de deep learning.
- Exemplos de vizinhos mais proximos para `k_nearest_neighbors`.

## Configuracao

Toda configuracao deve ficar em `config.yaml`, incluindo caminhos, datasets, pipelines, parametros de treino, parametros por modelo, configuracao de XAI e nomes das classes.

Exemplo:

```yaml
paths:
  raw_dataset_directory: dataset/raw
  processed_dataset_directory: dataset/processed
  machine_learning_run_directory: machine-learning/runs
  machine_learning_computer_configuration_file: machine-learning/my-computer.yaml
  documentation_directory: docs
  xai_output_directory: xai

training:
  epoch_count: 50
  early_stopping_patience: 8
  batch_size: 32
  image_size: 224
  random_seed: 42

pipelines:
  baseline:
    model_names:
      - dino
      - vgg16
      - efficientnet
      - resnet50
      - k_nearest_neighbors
    dataset_names:
      - sdss_raw
      - decals_raw
      - sdss_smote
      - decals_smote

  baseline_xai:
    model_names:
      - vgg16
      - resnet50
      - k_nearest_neighbors
    dataset_names:
      - sdss_smote
      - decals_smote
```

## Padroes de Codigo

- Todo codigo deve estar em ingles.
- Use nomes autoexplicativos.
- Evite siglas e abreviacoes que dificultem leitura.
- Mantenha funcoes pequenas e com responsabilidade unica.
- Separe cada modelo em seu proprio arquivo.
- Separe cada metodo de balanceamento em seu proprio arquivo.
- Centralize registros em arquivos `registry.py`.
- As CLIs devem ser interativas, com checkboxes ou listas de selecao para escolher datasets, modelos, pipelines e metodos.
- Use `mise-en-place` e `uv` como ferramentas padrao de ambiente e execucao.
- Mantenha o orquestrador raiz desacoplado das CLIs internas.

## Validacao Esperada

Comandos de ajuda:

```bash
uv run python main.py --help
uv run python dataset/main.py --help
uv run python machine-learning/main.py --help
uv run python xai/main.py --help
```

Cenarios minimos de teste:

- carregar e salvar H5 com chaves `images` e `ans`;
- aplicar cada metodo de balanceamento;
- aplicar multiplos balanceamentos em sequencia;
- executar uma pipeline curta com `k_nearest_neighbors`;
- gerar `machine-learning/my-computer.yaml` na primeira execucao da CLI de machine learning;
- criar uma run em `machine-learning/runs/`;
- criar relatorios em `docs/`;
- extrair amostras e gerar XAI em lote.

## Arquivos Grandes

Arquivos H5, checkpoints, runs, imagens geradas e outros artefatos grandes nao devem ser versionados no Git.
