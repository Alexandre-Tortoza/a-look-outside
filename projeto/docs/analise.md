# Benchmark por Modelo (SDSS raw)

**Resumo geral:** melhor `val_acc` foi **ResNet50 (90.08%)**; melhor tempo foi **EfficientNet-B0 (15.2 min)**.  
**Fonte:** somente `pesos/*/*.json` + logs em `docs/*/*.log` (ignorando `.pth`).

> Tempo por execução foi estimado no log do modelo: da primeira linha `Época 1/50` até a última linha do mesmo bloco.

## CNN

### Benchmark (runs encontrados)

| Execução | val_acc | Épocas | Timestamp JSON | Tempo no log |
|---|---:|---:|---|---:|
| `cnn_sdss_raw_ep50_20260501_133405` | 86.59% | 43 | 2026-05-01T14:32:45 | 121.7 min |
| `cnn_sdss_raw_ep50_20260501_153718` | 86.59% | 43 | 2026-05-01T16:34:54 | 65.5 min |

**Tempo total do modelo:** **187.2 min** (2 runs).

### Como rodou (código)

- Entrada: `main.py` -> `_acao_treinar` -> `modelos.cnn.treino.treinar`
- Pipeline: `modelos/cnn/treino.py` + engine `modelos/treinador.py` (treino em 1 estágio, AdamW + CosineAnnealingLR, checkpoint + curva + JSON)

### Parâmetros (melhor run no benchmark de tempo/acc)

`seed=42`, `epocas=50`, `batch_size=32`, `tamanho_imagem=224`, `lr=1e-3`, `paciencia_early_stop=10`, `scheduler_ativo=true`, `peso_decay=1e-4`, `dataset=sdss`, `versao_dataset=raw`.

### Resultado principal

**Melhor val_acc:** **86.59%** (`parou_cedo=false`, `epocas_totais=43`).

### Curva

![Curva CNN](./cnn/curva_treino.png)

---

## ResNet50

### Benchmark (runs encontrados)

| Execução | val_acc | Épocas | Timestamp JSON | Tempo no log |
|---|---:|---:|---|---:|
| `resnet50_sdss_raw_ep50_20260501_164415` | **90.08%** | 17 | 2026-05-01T17:02:34 | 28.7 min |

**Tempo total do modelo:** **28.7 min** (1 run).

### Como rodou (código)

- Entrada: `main.py` -> `_acao_treinar` -> `modelos.resnet50.treino.treinar`
- Pipeline: `modelos/resnet50/treino.py` + `modelos/_transfer_learning.py` (2 estágios: backbone congelado e depois descongelado)

### Parâmetros

`seed=42`, `epocas=50`, `batch_size=32`, `tamanho_imagem=224`, `lr_cabeca=1e-3`, `lr_backbone=1e-4`, `epocas_congelado=5`, `pretrained=true`, `paciencia_early_stop=10`, `scheduler_ativo=true`, `peso_decay=1e-4`, `dataset=sdss`, `versao_dataset=raw`.

### Resultado principal

**Melhor val_acc do benchmark:** **90.08%** (`parou_cedo=false`, `epocas_totais=17`).

### Curva

![Curva ResNet50](./resnet50/curva_treino.png)

---

## EfficientNet-B0

### Benchmark (runs encontrados)

| Execução | val_acc | Épocas | Timestamp JSON | Tempo no log |
|---|---:|---:|---|---:|
| `efficientnet_sdss_raw_ep50_20260501_171440` | 87.82% | 14 | 2026-05-01T17:23:11 | **15.2 min** |

**Tempo total do modelo:** **15.2 min** (1 run).

### Como rodou (código)

- Entrada: `main.py` -> `_acao_treinar` -> `modelos.efficientnet.treino.treinar`
- Pipeline: `modelos/efficientnet/treino.py` + `modelos/_transfer_learning.py` (2 estágios)

### Parâmetros

`seed=42`, `epocas=50`, `batch_size=32`, `tamanho_imagem=224`, `variante_backbone=efficientnet_b0`, `lr_cabeca=1e-3`, `lr_backbone=1e-4`, `epocas_congelado=5`, `pretrained=true`, `paciencia_early_stop=10`, `scheduler_ativo=true`, `peso_decay=1e-4`, `dataset=sdss`, `versao_dataset=raw`.

### Resultado principal

**val_acc:** **87.82%** (`parou_cedo=false`, `epocas_totais=14`).

### Curva

![Curva EfficientNet](./efficientnet/curva_treino.png)

---

## VGG16

### Benchmark (runs encontrados)

| Execução | val_acc | Épocas | Timestamp JSON | Tempo no log |
|---|---:|---:|---|---:|
| `vgg16_sdss_raw_ep50_20260501_173036` | 89.38% | 41 | 2026-05-01T19:56:05 | 164.7 min |

**Tempo total do modelo:** **164.7 min** (1 run).

### Como rodou (código)

- Entrada: `main.py` -> `_acao_treinar` -> `modelos.vgg16.treino.treinar`
- Pipeline: `modelos/vgg16/treino.py` + `modelos/_transfer_learning.py` (2 estágios)

### Parâmetros

`seed=42`, `epocas=50`, `batch_size=16`, `tamanho_imagem=224`, `lr_cabeca=1e-3`, `lr_backbone=1e-4`, `epocas_congelado=5`, `pretrained=true`, `paciencia_early_stop=10`, `scheduler_ativo=true`, `peso_decay=1e-4`, `dataset=sdss`, `versao_dataset=raw`.

### Resultado principal

**val_acc:** **89.38%** (`parou_cedo=false`, `epocas_totais=41`).

### Curva

![Curva VGG16](./vgg16/curva_treino.png)

---

## ViT

### Benchmark (runs encontrados)

| Execução | val_acc | Épocas | Timestamp JSON | Tempo no log |
|---|---:|---:|---|---:|
| `vit_sdss_raw_ep50_20260501_202722` | 85.55% | 37 | 2026-05-01T22:06:40 | 122.1 min |

**Tempo total do modelo:** **122.1 min** (1 run).

### Como rodou (código)

- Entrada: `main.py` -> `_acao_treinar` -> `modelos.vit.treino.treinar`
- Pipeline: `modelos/vit/treino.py` + `modelos/_transfer_learning.py` (2 estágios)

### Parâmetros

`seed=42`, `epocas=50`, `batch_size=32`, `tamanho_imagem=224`, `backbone=vit_base_patch16_224`, `lr_cabeca=1e-3`, `lr_backbone=1e-5`, `epocas_congelado=10`, `pretrained=true`, `paciencia_early_stop=10`, `scheduler_ativo=true`, `peso_decay=1e-4`, `dataset=sdss`, `versao_dataset=raw`.

### Resultado principal

**val_acc:** **85.55%** (`parou_cedo=false`, `epocas_totais=37`).

### Curva

![Curva ViT](./vit/curva_treino.png)

---

## DINO

### Benchmark (runs encontrados)

| Execução | val_acc | Épocas | Timestamp JSON | Tempo no log |
|---|---:|---:|---|---:|
| `dino_sdss_raw_ep50_20260502_144916` | 64.13% | 5 | 2026-05-02T15:08:04 | 39.7 min |
| `dino_sdss_raw_ep50_20260502_153436` | 82.71% | 24 | 2026-05-02T17:04:09 | 93.6 min |

**Tempo total do modelo:** **133.3 min** (2 runs).

### Como rodou (código)

- Entrada: `main.py` -> `_acao_treinar` -> `modelos.dino.treino.treinar`
- Pipeline específico: `modelos/dino/treino.py` -> `modelos/dino/ajuste_fino.py` -> `modelos/_transfer_learning.py`
- Modo usado no JSON: `modo=hub` (sem pré-treino self-supervisionado nesta execução)

### Parâmetros (melhor run)

`seed=42`, `epocas=50`, `batch_size=32`, `tamanho_imagem=224`, `modo=hub`, `backbone=dinov2_vitb14`, `lr_cabeca=1e-3`, `lr_backbone=1e-5`, `epocas_congelado=5`, `epocas_pre_treino=100`, `epocas_ajuste_fino=30`, `tamanho_projecao=65536`, `temperatura_professor=0.04`, `temperatura_aluno=0.1`, `paciencia_early_stop=10`, `scheduler_ativo=true`, `peso_decay=1e-4`, `dataset=sdss`, `versao_dataset=raw`.

### Resultado principal

**Melhor val_acc:** **82.71%** (`parou_cedo=false`, `epocas_totais=24`).

### Curva

Curva não encontrada em `docs/dino/curva_treino.png` nesta versão.

---

## Fechamento do benchmark

- **Total de runs:** 8  
- **Tempo total acumulado:** **651.3 min** (~10h51min)  
- **Ranking por acurácia:** ResNet50 > VGG16 > EfficientNet > CNN > ViT > DINO
