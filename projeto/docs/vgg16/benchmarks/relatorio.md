# Benchmark VGG16 × Datasets

**Modelos avaliados:** T1 (SDSS/raw), T2 (SDSS/hibrido), T3 (DECaLS/raw), T4 (fusao), T5 (fine-tune T2→fusao)  
**Datasets de avaliacao:** SDSS, DECaLS, fusao  
**Modo split:** test split 15% (in-distribution)  
**Modo full:** dataset completo (cross-dataset generalization)  

## Treinos (in-distribution — val_acc no test split)

| ID | Descricao | Dataset treino | val_acc | F1 macro |
|---|---|---|---:|---:|
| **t1** | VGG16 treina SDSS/raw | sdss/raw | **0.8838** | 0.7600 |
| **t2** | VGG16 treina SDSS/hibrido | sdss/hibrido | **0.9679** | 0.9677 |
| **t3** | VGG16 treina DECaLS/raw | decals/raw | **0.8512** | 0.8268 |
| **t4** | VGG16 treina fusao | fusao/raw | **0.8689** | 0.8592 |
| **t5** | Fine-tune T2 → fusao (estado da arte) | fusao/raw | **0.8689** | 0.8592 |

## Matriz Cross-Dataset (acuracia)

> Diagonal = in-distribution (test split). Fora = cross-dataset (dataset completo).

| Treino \ Eval | **SDSS** | **DECALS** | **FUSAO** |
|---|---:|---:|---:|
| **SDSS/raw** | **0.8838** | 0.0520 | 0.5471 |
| **SDSS/hibrido ★** | **0.9679** | 0.0843 | 0.5744 |
| **DECaLS/raw** | 0.0638 | **0.8512** | 0.4459 |
| **fusao** | 0.9162 | 0.9071 | **0.8689** |
| **FT→fusao ⚡** | 0.9162 | 0.9071 | **0.8689** |

## Analise de Generalizacao (gap in-dist vs cross)

| Treino | In-dist acc | Melhor cross acc | Gap |
|---|---:|---:|---:|
| t1 (sdss/raw) | 0.8838 | 0.5471 | +0.3367 |
| t2 (sdss/hibrido) | 0.9679 | 0.5744 | +0.3935 |
| t3 (decals/raw) | 0.8512 | 0.4459 | +0.4053 |
| t4 (fusao/raw) | 0.8689 | 0.9162 | -0.0472 |
| t5 (fusao/raw) | 0.8689 | 0.9162 | -0.0472 |

## Impacto do Fine-Tuning (T5 vs T2)

| Avaliacao | T2 (SDSS/hibrido) | T5 (FT→fusao) | Delta |
|---|---:|---:|---:|
| SDSS completo | 0.9679 | 0.9162 | **-0.0517** |
| DECaLS completo | 0.0843 | 0.9071 | **+0.8228** |
| fusao test split | 0.5744 | 0.8689 | **+0.2945** |

## Conclusao

- **Melhor dataset in-distribution:** `t2` (acc=0.9679)
- **Melhor cross-dataset:** treinado em `t4`, avaliado como `t4_c7` (acc=0.9162)
- Detalhes por benchmark: `docs/vgg16/benchmarks/*.md`
