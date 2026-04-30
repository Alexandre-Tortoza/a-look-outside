```markdown
# ideia.md — Arquitetura do Projeto

## Visão Geral

Repositório para artigo acadêmico sobre visão computacional aplicada à classificação morfológica de galáxias. O objetivo central é comparar abordagens de modelos, técnicas de pré-processamento, balanceamento e aumento de dados sobre os datasets Galaxy10 SDSS e Galaxy10 DECaLS.

---

## Princípios de Design

- **Isolamento total**: cada modelo, técnica ou experimento vive no seu próprio módulo
- **Reprodutibilidade**: todo modelo treinado salva `.pth` + configuração usada
- **Configuração separada**: parâmetros em `config.py`, nunca hardcoded no treino
- **Explicabilidade opcional**: XAI pode ser habilitado por flag em qualquer modelo
- **Nomenclatura descritiva**: arquivos e pastas em `snake_case`, pt-br

---

## Estrutura de Diretórios

```

projeto/
│
├── ideia.md
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── raw/
│   │   ├── sdss/
│   │   │   └── galaxy10_sdss.h5
│   │   └── decals/
│   │       └── galaxy10_decals.h5
│   │
│   └── processados/
│       ├── sdss_aumento_de_dados.h5
│       ├── sdss_balanceado_smote.h5
│       ├── decals_aumento_de_dados.h5
│       ├── decals_balanceado_smote.h5
│       └── ... (outros experimentos)
│
├── pre_processamento/
│   ├── aumento_de_dados.py       # data augmentation
│   ├── balanceamento.py          # SMOTE, oversampling, etc.
│   ├── normalizacao.py
│   ├── divisao_treino_teste.py
│   └── config.py                 # parâmetros de pré-processamento
│
├── modelos/
│   │
│   ├── cnn/
│   │   ├── config.py             # hiperparâmetros, paths, flags
│   │   ├── modelo.py             # definição da arquitetura
│   │   ├── treino.py             # loop de treino
│   │   ├── execucao.py           # carrega .pth e roda inferência
│   │   ├── avaliacao.py          # métricas, matriz de confusão
│   │   └── xai.py                # Grad-CAM (habilitado via config)
│   │
│   ├── resnet50/
│   │   ├── config.py
│   │   ├── modelo.py
│   │   ├── treino.py
│   │   ├── execucao.py
│   │   ├── avaliacao.py
│   │   └── xai.py                # Grad-CAM
│   │
│   ├── efficientnet/
│   │   ├── config.py
│   │   ├── modelo.py
│   │   ├── treino.py
│   │   ├── execucao.py
│   │   ├── avaliacao.py
│   │   └── xai.py
│   │
│   ├── vgg16/
│   │   ├── config.py
│   │   ├── modelo.py
│   │   ├── treino.py
│   │   ├── execucao.py
│   │   ├── avaliacao.py
│   │   └── xai.py
│   │
│   ├── vit/                      # Vision Transformer
│   │   ├── config.py
│   │   ├── modelo.py
│   │   ├── treino.py
│   │   ├── execucao.py
│   │   ├── avaliacao.py
│   │   └── xai.py                # Attention rollout
│   │
│   ├── dino/                     # self-supervised (DINO v1/v2)
│   │   ├── config.py
│   │   ├── modelo.py             # backbone + cabeça de projeção
│   │   ├── pre_treino.py         # loop self-supervised (destilação)
│   │   ├── ajuste_fino.py        # fine-tuning supervisionado
│   │   ├── execucao.py
│   │   ├── avaliacao.py
│   │   └── xai.py                # Attention maps (patch tokens)
│   │
│   └── multimodal/               # imagem + metadados tabulares (DECaLS)
│       ├── config.py
│       ├── modelo.py             # fusão CNN + MLP tabular
│       ├── treino.py
│       ├── execucao.py
│       ├── avaliacao.py
│       └── xai.py                # Grad-CAM (visual) + SHAP (tabular)
│
├── pesos/                        # modelos treinados (.pth)
│   ├── cnn/
│   ├── resnet50/
│   ├── efficientnet/
│   ├── vgg16/
│   ├── vit/
│   ├── dino/
│   └── multimodal/
│
├── docs/
│   ├── cnn/
│   │   ├── resultados.md
│   │   ├── curva_treino.png
│   │   └── matriz_confusao.png
│   ├── resnet50/
│   ├── efficientnet/
│   ├── vgg16/
│   ├── vit/
│   ├── dino/
│   └── multimodal/
│
└── utils/
    ├── logger.py                 # logging padronizado
    ├── metricas.py               # acurácia, F1, top-k
    ├── visualizacao.py           # plots reutilizáveis
    └── reproducibilidade.py      # seed global, determinismo

```

---

## Fluxo de Trabalho

### Modelos supervisionados

```

raw/ → pre_processamento/ → processados/
                                  ↓
                          modelos/<modelo>/treino.py
                                  ↓
                          pesos/<modelo>/<experimento>.pth
                                  ↓
                    execucao.py + avaliacao.py + xai.py
                                  ↓
                          docs/<modelo>/resultados.md

```

### DINO (self-supervised)

```

raw/ → pre_processamento/ → processados/
                                  ↓
                      modelos/dino/pre_treino.py     # destilação professor-aluno
                                  ↓
                      pesos/dino/<experimento>_pretreino.pth
                                  ↓
                      modelos/dino/ajuste_fino.py    # fine-tuning supervisionado
                                  ↓
                      pesos/dino/<experimento>_ajuste_fino.pth
                                  ↓
                    execucao.py + avaliacao.py + xai.py
                                  ↓
                          docs/dino/resultados.md

```

---

## Controle por config.py

Cada `config.py` expõe ao menos:

```python
# exemplo — modelos/resnet50/config.py

DATASET         = "decals"           # "sdss" | "decals"
VERSAO_DATASET  = "aumento_de_dados" # nome do .h5 em processados/
EPOCAS          = 50
BATCH_SIZE      = 32
LR              = 1e-4
SEED            = 42
SALVAR_PESOS    = True
HABILITAR_XAI   = False              # liga/desliga xai.py
```

O DINO inclui flags adicionais:

```python
# exemplo — modelos/dino/config.py

EPOCAS_PRE_TREINO    = 100
EPOCAS_AJUSTE_FINO   = 30
TAMANHO_PROJECAO     = 65536
TEMPERATURA_PROFESSOR = 0.04
TEMPERATURA_ALUNO    = 0.1
HABILITAR_XAI        = False
```

---

## XAI por Tipo de Modelo

| Modelo                              | Técnica Visual                | Técnica Tabular |
|-------------------------------------|-------------------------------|-----------------|
| CNN / ResNet / EfficientNet / VGG16 | Grad-CAM                      | —               |
| ViT                                 | Attention Rollout             | —               |
| DINO                                | Attention Maps (patch tokens) | —               |
| Multimodal                          | Grad-CAM (imagem)             | SHAP (tabular)  |

XAI sempre salva outputs em `docs/<modelo>/xai/`.

---

## Experimentos Planejados

### Modelos (baseline → avançado)

1. CNN simples (baseline próprio)
2. VGG16
3. ResNet-50
4. EfficientNet-B0
5. ViT / CvT
6. DINO (self-supervised) → fine-tuning no Galaxy10
7. Multimodal (CNN + MLP tabular) — gap inédito no Galaxy10 DECaLS

### Variações de Dataset

- Sem pré-processamento (controle)
- Com aumento de dados
- Com balanceamento (SMOTE / oversampling)
- Combinações

### Referência a Bater

- **Astroformer**: ~94,86% top-1 no Galaxy10 DECaLS

---

## Convenções

- Código: `snake_case`, português
- Um experimento = um `.pth` nomeado descritivamente
  - ex: `resnet50_decals_aumento_epoca50.pth`
  - ex: `dino_decals_pretreino_epoca100.pth`
  - ex: `dino_decals_ajuste_fino_epoca30.pth`
- Logs e plots sempre em `docs/<modelo>/`
- Nenhum hiperparâmetro hardcoded fora do `config.py`

```
