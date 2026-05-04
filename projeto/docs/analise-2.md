# Analise-2: Top 3 Modelos — Visao Geral Completa (raw + hibrido)

**Escopo:** todos os runs encontrados nos 3 modelos de melhor desempenho global.
**Fonte:** `pesos/*/*.json` + logs em `docs/*/*.log` (`.pth` ignorados).
**Nota sobre `epocas_totais` nos JSONs:** o campo registra a _época do melhor checkpoint_ (não o total de épocas executadas). O total real é extraído do log.

---

## Ranking geral (melhor run por modelo)

| # | Modelo | Dataset | val_acc | Época melhor ckpt | Epocas treinadas | Tempo |
|---|--------|---------|--------:|------------------:|-----------------:|------:|
| 1 | VGG16 | hibrido | **96.70%** | 47 | 50 | 533.7 min |
| 2 | ResNet50 | hibrido | **96.45%** | 50 | 50 | 189.2 min |
| 3 | EfficientNet-B0 | raw | **87.82%** | 14 | 24* | 15.7 min |

\* parou por early stopping.

Delta hibrido vs raw: VGG16 **+7.32 pp** (89.38 → 96.70), ResNet50 **+6.37 pp** (90.08 → 96.45).
EfficientNet nao tem run hibrido — posicao no ranking pode mudar.

---

## 1. VGG16

### Todos os runs

| Execucao | Dataset | val_acc | Época melhor | Epocas treinadas | Tempo (log) |
|---|---|---:|---:|---:|---:|
| `vgg16_sdss_raw_ep50_20260501_173036` | raw | 89.38% | 41 | 50 | 166.8 min |
| `vgg16_sdss_hibrido_ep50_20260502_221730` | hibrido | **96.70%** | 47 | 50 | 533.7 min |

**Tempo total acumulado:** 700.5 min (~11h41min, 2 runs).

### Parametros

| Param | raw | hibrido |
|---|---|---|
| `batch_size` | 16 | 16 |
| `lr_cabeca` | 1e-3 | 1e-3 |
| `lr_backbone` | 1e-4 | 1e-4 |
| `epocas_congelado` | 5 | 5 |
| `paciencia_early_stop` | 10 | 5 |
| `label_smoothing` | — | 0.05 |
| `distilacao_temperatura` | — | 4.0 |
| `distilacao_alpha` | — | 0.7 |

Diferencas relevantes no run hibrido: `paciencia_early_stop` reduzida de 10 para 5, adicao de `label_smoothing=0.05` e destilacao de conhecimento (`alpha=0.7`, `temperatura=4.0`).

### Convergencia (run hibrido)

- Stage 1 (backbone congelado, 5 epocas): vai de 75.70% a 81.92% val_acc.
- Stage 2 (backbone liberado, ep6-50): salto imediato para 82.82% na ep6, progresso constante.
- Melhor na ep47 (96.70%); ep48-50 ficam em 96.62-96.67% — plateau quase atingido.
- Nao disparou early stop (paciencia=5) porque continuou melhorando lentamente ate ep47.

| Etapa | Epoca | val_acc |
|---|---:|---:|
| Stage 2 inicio | 6 | 82.82% |
| Meio do treino | 25 | 94.67% |
| Melhor | 47 | 96.70% |
| Fim (ep50) | 50 | 96.62% |

### Curvas

![Curva VGG16](./vgg16/curva_treino.png)

---

## 2. ResNet50

### Todos os runs

| Execucao | Dataset | val_acc | Época melhor | Epocas treinadas | Tempo (log) |
|---|---|---:|---:|---:|---:|
| `resnet50_sdss_raw_ep50_20260501_164415` | raw | 90.08% | 17 | 27* | 29.3 min |
| `resnet50_sdss_hibrido_ep50_20260502_190810` | hibrido | **96.45%** | 50 | 50 | 189.2 min |

\* early stop apos ep27 (sem melhora desde ep17).

**Tempo total acumulado:** 218.5 min (~3h39min, 2 runs).

### Parametros

| Param | raw | hibrido |
|---|---|---|
| `batch_size` | 32 | 32 |
| `lr_cabeca` | 1e-3 | 1e-3 |
| `lr_backbone` | **1e-4** | **5e-5** |
| `epocas_congelado` | 5 | 5 |
| `label_smoothing` | — | 0.1 |

Diferenca principal: `lr_backbone` foi reduzido pela metade (1e-4 → 5e-5) e adicionado `label_smoothing=0.1`. Essa combinacao impediu early stop e manteve o modelo melhorando ate a ep50.

### Convergencia (run hibrido)

- Stage 2 inicio (ep6): 77.12% val_acc.
- Progressao regular sem overfitting — loss de treino descendo junto com val_acc subindo.
- Melhor na ep50 (96.45%): o modelo ainda estava melhorando ao final, indicando potencial para mais epocas.

| Etapa | Epoca | val_acc |
|---|---:|---:|
| Stage 2 inicio | 6 | 77.12% |
| Meio do treino | 25 | 95.60% |
| Fim = melhor | 50 | 96.45% |

### Convergencia (run raw — destaque)

O run raw e notavel pela velocidade: Stage 2 iniciou na ep6 e o melhor resultado (90.08%) foi atingido ja na ep17, em apenas ~12 min de Stage 2. Early stop na ep27.

### Curvas

![Curva ResNet50](./resnet50/curva_treino.png)

---

## 3. EfficientNet-B0

### Todos os runs

| Execucao | Dataset | val_acc | Época melhor | Epocas treinadas | Tempo (log) |
|---|---|---:|---:|---:|---:|
| `efficientnet_sdss_raw_ep50_20260501_171440` | raw | **87.82%** | 14 | 24* | 15.7 min |

\* early stop apos ep24 (sem melhora desde ep14).
Nao ha run com dataset hibrido.

**Tempo total acumulado:** 15.7 min (1 run).

### Parametros

`seed=42`, `epocas=50`, `batch_size=32`, `tamanho_imagem=224`, `variante_backbone=efficientnet_b0`, `lr_cabeca=1e-3`, `lr_backbone=1e-4`, `epocas_congelado=5`, `pretrained=true`, `paciencia_early_stop=10`, `scheduler_ativo=true`, `peso_decay=1e-4`.

### Convergencia

O modelo mais eficiente em tempo: cada epoca do Stage 2 leva ~45 segundos.

- Stage 2 inicio (ep6): 78.60% val_acc — salto maior que ResNet50 e VGG16 na mesma etapa.
- Melhor rapidamente na ep14 (87.82%); sinais de overfitting aparecem a partir da ep15 (val_acc cai enquanto train_acc sobe).
- Early stop na ep24.

| Etapa | Epoca | val_acc |
|---|---:|---:|
| Stage 2 inicio | 6 | 78.60% |
| Melhor | 14 | 87.82% |
| Early stop | 24 | 86.62% |

### Curvas

![Curva EfficientNet](./efficientnet/curva_treino.png)

---

## Comparativo direto — Top 3

### Performance

| Modelo | Melhor run | val_acc | Epocas ate melhor | Tempo total treino |
|---|---|---:|---:|---:|
| VGG16 | hibrido | **96.70%** | 47 | 700.5 min |
| ResNet50 | hibrido | 96.45% | 50 | 218.5 min |
| EfficientNet | raw | 87.82% | 14 | 15.7 min |

### Eficiencia (acc / tempo)

| Modelo | acc melhor | Tempo run melhor | Acc/min |
|---|---:|---:|---:|
| EfficientNet | 87.82% | 15.7 min | **5.59 pp/min** |
| ResNet50 hibrido | 96.45% | 189.2 min | 0.51 pp/min |
| VGG16 hibrido | 96.70% | 533.7 min | 0.18 pp/min |

EfficientNet lidera em eficiencia; ResNet50 tem melhor relacao custo-beneficio quando se precisa de alta acuracia.

### Comportamento de convergencia no Stage 2

| Modelo | val_acc Stage 2 ep1 | val_acc melhor | Delta Stage 2 |
|---|---:|---:|---:|
| EfficientNet | 78.60% | 87.82% | +9.22 pp |
| ResNet50 raw | 74.47% | 90.08% | +15.61 pp |
| ResNet50 hibrido | 77.12% | 96.45% | +19.33 pp |
| VGG16 raw | 82.46% | 89.38% | +6.92 pp |
| VGG16 hibrido | 82.82% | 96.70% | +13.88 pp |

### Impacto do dataset hibrido

Os dois modelos com runs hibridos mostraram ganhos expressivos e consistentes:

- ResNet50: 90.08% → 96.45% (+6.37 pp)
- VGG16: 89.38% → 96.70% (+7.32 pp)

Isso sugere que o dataset hibrido e o fator principal do ganho, independente das mudancas de hiperparametros. EfficientNet e candidato natural para o proximo run hibrido.

O DINO hibrido, por outro lado, falhou completamente (val_acc presa em ~10%, baseline de 10 classes) nos dois runs de 2026-05-03 — indicando problema de configuracao especifico ao pipeline DINO com o dataset hibrido (provavelmente incompatibilidade na cabeca MLP com `cabeca_mlp=true` ou problema no carregamento do dataset).

---

## Fechamento

- **Total de runs (top 3):** 5 runs validos (1 EfficientNet, 2 ResNet50, 2 VGG16)
- **Ranking definitivo por val_acc:** VGG16 (96.70%) > ResNet50 (96.45%) > EfficientNet (87.82%)
- **Ranking por eficiencia:** EfficientNet >> ResNet50 >> VGG16
- **Proximo passo natural:** run EfficientNet com dataset hibrido (estimativa: ~30-50 min baseado na velocidade por epoca x maior dataset).
