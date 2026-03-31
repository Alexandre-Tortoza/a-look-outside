# A Look Outside
## Classificação Multi-modal de Galáxias com Federated Learning e Explicabilidade

---

## 🎯 Visão Geral do Projeto

Um projeto de pesquisa que integra **três pilares inovadores** para classificação de morfologia galáctica:

1. **Aprendizado Federado (FL):** Cada observatório compartilha apenas os pesos dos modelos treinados localmente, preservando a privacidade operacional e política dos dados brutos.

2. **Abordagem Multi-modal:** Combinação de **imagens** (morfologia visual) com **metadados físicos** (cor, massa estelar, redshift) para aumentar robustez em ambientes heterogêneos de dados.

3. **Explicabilidade (XAI):** Identificação clara de quais anomalias galácticas residem na **aparência visual**, nos **metadados**, ou em alguma **combinação dos dois**, trazendo insights físicos através de padrões aprendidos.

---

## 📋 Problema de Pesquisa e Relevância

### Contexto Astronômico

Com o advento de levantamentos astronômicos modernos, como o **Vera Rubin Observatory (LSST)** e o **Euclid (ESA)**, espera-se catalogar bilhões de galáxias na próxima década. A classificação morfológica — _espiral, elíptica, espiral barrada, em fusão_ — é fundamental para compreender a formação e evolução galáctica, mas a inspeção manual é humanamente inviável nessa escala.

Projetos de ciência cidadã, como o **Galaxy Zoo**, mobilizaram centenas de milhares de voluntários, mas ainda assim cobrem apenas uma fração dos dados disponíveis.

### Desafio Operacional: Dados Gigantescos e Heterogêneos

Os **dados coletados pelos observatórios são gigantescos** (centenas de terabytes). Além disso, existem **questões políticas e operacionais** que impedem o compartilhamento de dados brutos entre instituições. Cada observatório precisa de autonomia para manter seus dados locais, ao mesmo tempo em que deseja se beneficiar de colaboração global em modelos.

**Federated Learning é a solução:** Cada nó (observatório) treina seu modelo localmente e compartilha apenas os pesos com um servidor central. Esses pesos são agregados para gerar um modelo global, sem nunca mover os dados brutos.

### Desafio Técnico: Heterogeneidade de Dados

Porém, dados **não-heterogêneos** (non-IID) degradam severamente o aprendizado federado. Cada observatório produz imagens com resolução, profundidade, e fotometria diferentes. Isso causa **deriva de dados** (data drift) que quebra as suposições de convergência dos algoritmos clássicos como FedAvg.

**Nossa hipótese:** A abordagem **multi-modal** ajuda nisso. Metadados físicos (cor, massa estelar) sofrem **menos variação** entre observatórios do que imagens brutas, criando um **âncora consistente** que estabiliza o treinamento federado mesmo com imagens heterogêneas.

### Oportunidade de Pesquisa: Explicabilidade em Física

Há evidências de que **XAI (Explainable AI) revelou novos fenômenos físicos** ao identificar padrões imperceptíveis aos humanos. Nosso objetivo é trazer essa ideia para **classificação de galáxias anômalas:** usar explicabilidade para separar anomalias que residem na **morfologia visual** (ex: artefatos de processamento) vs. anomalias nos **metadados** (ex: galáxias com propriedades físicas inconsistentes).

---

## 🔬 Hipóteses e Objetivos

### Hipóteses Principais

1. **Hipótese Multi-modal:** Modelos que combinam imagens + metadados tabulares apresentam **maior robustez em ambientes federados heterogêneos** comparado a modelos unimodais (apenas imagem).

2. **Hipótese de Heterogeneidade:** Dados não-IID entre observatórios causam degradação significativa em FL clássico (FedAvg), mas essa degradação é **atenuada pela consistência de metadados físicos**.

3. **Hipótese de Explicabilidade:** Padrões anômalos em galáxias podem ser **decompostos em contribuições de modalidades** (visual vs. tabular), revelando anomalias físicas que não são visualmente óbvias.

4. **Hipótese Baseline:** Arquiteturas pré-treinadas em ImageNet (EfficientNet, ViT) superam modelos treinados do zero no benchmark **Galaxy10 DECaLS**, e modelos leves como MobileNet atingem acurácia competitiva com tempo de treinamento significativamente menor.

### Objetivos Específicos

**Fase 1 — Baseline Centralizado:**
- Implementar e comparar **3+ arquiteturas** (ResNet-50, EfficientNet-B3, ViT-B/16) no Galaxy10 DECaLS.
- Integrar **features tabulares** e validar ganho multi-modal em configuração centralizada.
- Analisar impacto de **augmentação específica para imagens astronômicas** (rotações, ruído Poisson).

**Fase 2 — Federated Learning:**
- Implementar **FedAvg** e **FedProx** com dados heterogêneos (simulando múltiplos observatórios).
- Medir degradação de acurácia em cenários non-IID.
- Demonstrar que **FL + multi-modal** mitiga essa degradação vs. FL + unimodal.

**Fase 3 — Explicabilidade:**
- Aplicar **Grad-CAM** para imagens e **Feature Attribution** para metadados tabulares.
- Desenvolver método para **decompor contribuições por modalidade** em anomalias galácticas.
- Validar que insights extraídos são fisicamente significativos.

---

## 🏗️ Metodologia

### Fase 1: Baseline Centralizado (Aprendizado Supervisionado Multi-modal)

#### **Camada 1 — Imagens (Modalidade Visual)**

**Dataset:** Galaxy10 DECaLS (17.736 imagens, 10 classes, 256×256 px)

- **Resolução:** 256×256 px, 3 bandas (g, r, i)
- **Classes:** 10 morfologias de galáxias
- **Divisão:** 70/15/15 (treino/validação/teste) com estratificação por classe
- **Acesso:** [zenodo.org/records/10845026](https://zenodo.org/records/10845026)

**Pré-processamento:**
- Normalização por média e desvio padrão do dataset
- Redimensionamento para 224×224 px
- **Augmentação específica para astronomia:**
  - Rotação aleatória de 0–360° (galáxias não têm orientação preferencial)
  - Flip horizontal/vertical
  - Zoom ±10%
  - Ruído Poisson (simulando background astronômico)

#### **Camada 2 — Features Tabulares (Modalidade Numérica)**

Integração de **metadados físicos** para criar setup **multi-modal**:

| Fonte | Features Principais | Volume | Acesso |
|-------|------------------|--------|--------|
| **Galaxy Zoo DECaLS Catálogos** | Redshift (`z`), Magnitudes (u,g,r,i,z), Cor (g-r), Raio efetivo, Elipticidade | 314K galáxias | [zenodo.org/records/4196267](https://zenodo.org/records/4196267) |
| **NASA-Sloan Atlas (NSA)** | Magnitude absoluta, **Massa estelar**, Redshift, Elipticidade | Incluso em GZ | [Hugging Face](https://huggingface.co/datasets/MultimodalUniversity/GalaxyZoo2) |
| **SDSS DR17** | Parâmetros espectroscópicos, Dispersão de velocidade, Flags de qualidade | 6.7 GB (bulk) | [skyserver.sdss.org/casjobs](https://skyserver.sdss.org/casjobs) |
| **MPA-JHU Properties** | Taxa de formação estelar (SFR), **Metalicidade**, Índices espectrais | ~2M espectros | [NOIRLab MPA-JHU](https://www.noirlab.edu/science/surveys/sdss/spectral-classification) |

**Por que metadados físicos ajudam em FL:** 
- Imagens variam drasticamente entre observatórios (resolução, profundidade, PSF)
- Metadados físicos (massa, cor) são **derivados de princípios que independem do instrumento**
- Isso cria um **sinal consistente** mesmo com imagens heterogêneas

#### **Modelos Baseline (Unimodal)**

- **(a) CNN customizada** (baseline)
- **(b) ResNet-50** fine-tuned
- **(c) EfficientNet-B3** fine-tuned
- **(d) ViT-B/16** fine-tuned

**Treinamento:**
- Otimizador: Adam
- Learning Rate Scheduler: Cosine annealing com warm-up
- Early stopping: paciência de 10 epochs
- Class weights inversamente proporcionais à frequência
- Métricas: Acurácia, F1 macro, AUC-ROC

#### **Modelos Multi-modal**

**Late Fusion Architecture:**

```
Imagens → CNN/ViT → Feature Vector (2048D)    |
                                                ├→ Concatenação → MLP → Classificação
Features Tabulares → Embedding Tabular → FV  |
                                (128D)
```

- Extrator visual: Backbone pré-treinado
- Extrator tabular: MLP de 2 camadas (normalização com StandardScaler)
- Fusion: Concatenação simples
- Head classifier: MLP (512 → 256 → 10)

**Justificativa de Late Fusion:**
- Cada modalidade treina seu extrator independentemente
- Simples de implementar e debugar
- Natural para dados heterogêneos

#### **Interpretabilidade: Baseline**

Aplicar **Grad-CAM** para entender quais regiões das imagens guiam a classificação de cada modelo.

---

### Fase 2: Federated Learning com Dados Heterogêneos

#### **Simulação de Nós Federados**

Particionamos datasets para simular heterogeneidade não-IID realista:

| Nó | Dataset | Heterogeneidade | Simulação Real |
|---|---|---|---|
| **A** | Galaxy10 SDSS | Resolução: 69×69 px (0.396″/px), Survey raso | Legacy observatório com hardware antigo |
| **B** | GZ DECaLS Campaign A | Footprint sul, ~5 labels/galáxia | Observatório com cobertura geográfica limitada |
| **C** | GZ DECaLS Campaign C | Footprint diferente, ~30 labels/galáxia | Observatório com alta qualidade de anotações |
| **D** | GZ DECaLS z > 0.1 | Objetos com alto redshift (distantes) | Observatório especializado em galáxias distantes |

**Tipos de Heterogeneidade Simulada:**
- ✅ Diferenças de resolução espacial (0.396″ vs. 0.262″)
- ✅ Diferenças de profundidade (r=22.2 vs. r=23.6)
- ✅ Distribuições diferentes de redshift
- ✅ Qualidade heterogênea de labels
- ✅ Footprints geográficos distintos

Isso reproduz cenários realistas que algoritmos como **FedAvg** e **FedProx** precisam lidar.

#### **Algoritmos Federados a Implementar**

1. **FedAvg** (Baseline clássico)
   - Agregação simples de gradientes
   - Esperado: degradação severa em non-IID

2. **FedProx** (Mitigação de non-IID)
   - Adiciona termo proximal μ na função objetivo local
   - Esperado: melhora em convergência e acurácia

3. **FedAvg + Multi-modal** (Nossa Contribuição)
   - FedAvg aplicado apenas no **head classifier**
   - Extractores visuais e tabulares treinam localmente
   - Hipótese: Consistência de metadados estabiliza agregação

#### **Configuração Experimental**

- **Rodadas de comunicação:** 50–100
- **Epochs por cliente por rodada:** 5
- **Batch size local:** 32
- **Learning rate:** 0.001 com decay exponencial
- **Fração de clientes por rodada:** 1.0 (todos os nós participam)

#### **Métricas Federadas**

- **Acurácia global** (teste centralizado com modelo agregado)
- **Divergência de modelos** (distância L2 entre pesos locais e globais)
- **Comunicação total** (número de parâmetros transmitidos)
- **Convergência** (curva de acurácia vs. rodadas)

---

### Fase 3: Explicabilidade (XAI) para Anomalias

#### **Objetivo**

Classificar galáxias **anômalas** e atribuir causas:
- Anomalia **visual:** Morfologia estranha (fusões, distorções)
- Anomalia **tabular:** Propriedades físicas inconsistentes (p.ex., galáxia massiva com redshift alto incomum)
- Anomalia **conjunta:** Combinação de ambas

#### **Métodos de Explicabilidade**

**Para Modalidade Visual:**
- **Grad-CAM:** Heatmaps de regiões importantes
- **LIME:** Explicações locais com modelos simples

**Para Modalidade Tabular:**
- **SHAP:** Importância de cada feature
- **Permutation Feature Importance**

**Integração Multi-modal:**
- Comparar contribuições de Grad-CAM (visual) vs. SHAP (tabular) para mesma amostra
- Se Grad-CAM indica anomalia em região específica mas SHAP indica metadados normais → anomalia visual
- Se Grad-CAM normal mas SHAP mostra desvio de múltiplos metadados → anomalia tabular

#### **Dataset de Anomalias**

Selecionar ~500 galáxias com:
- Baixa confiança do modelo (entropia alta)
- Alto erro de classificação
- Metadados estatisticamente outliers (IQR > 3σ)

Validar interpretações com astrônomos ou Galaxy Zoo volunteers.

---

## 📊 Datasets e Acesso

### Resumo de Acesso aos Dados

| Dataset | Tamanho | Formato | Link |
|---|---|---|---|
| **Galaxy10 DECaLS** | ~1.4 GB | .h5 | [zenodo.org/records/10845026](https://zenodo.org/records/10845026) |
| **Galaxy10 SDSS** | ~600 MB | .h5 | [zenodo.org/records/10844811](https://zenodo.org/records/10844811) |
| **GZ DECaLS Catálogos** | ~50 MB | .csv | [zenodo.org/records/4196267](https://zenodo.org/records/4196267) |
| **SDSS DR17 NSA Features** | — | .csv (incluso) | Incluso nos catálogos acima |
| **SDSS CasJobs (SQL)** | on-demand | Query SQL | [skyserver.sdss.org/casjobs](https://skyserver.sdss.org/casjobs) |
| **MPA-JHU Properties** | — | Query/Download | [NOIRLab Database](https://www.noirlab.edu/science/surveys/sdss/spectral-classification) |

### Fluxo de Trabalho Recomendado

1. **Início:** Começar com **Galaxy10 DECaLS** (arquivo único, 1.4 GB, carrega em 3 linhas de Python)
2. **Expansão:** Adicionar **Galaxy10 SDSS** como Nó A para simular heterogeneidade
3. **Features:** Cruzar com catálogos CSV do Galaxy Zoo DECaLS para features tabulares (NSA)
4. **Escalabilidade:** Usar CasJobs ou download bulk do SDSS DR17 para análises deeper
5. **Federated:** Particionar os dados conforme seção de simulação de nós para experimentos de aprendizado federado

---

## 🛠️ Ferramentas e Stack Tecnológico

### Dependências Principais

**Processamento e ML:**
- `PyTorch` — Framework principal para deep learning
- `torchvision` — Pre-trained models e data augmentation
- `timm` (PyTorch Image Models) — Modelos adicionais (EfficientNet, ViT)

**Interpretabilidade:**
- `Captum` — Grad-CAM e attribution methods
- `LIME` — Explicações locais
- `SHAP` — Feature importance para dados tabulares

**Federated Learning:**
- `Flower` — Framework de simulação de FL
- `PySyft` — Alternativa para privacidade diferencial

**Processamento de Dados:**
- `pandas`, `numpy` — Manipulação de dados tabulares
- `h5py` — Leitura de datasets .h5
- `astropy` — Utilities astronômicas

**Visualização e Análise:**
- `matplotlib`, `seaborn` — Gráficos e heatmaps
- `scikit-learn` — Métricas e pré-processamento

### Infraestrutura

- **Ambiente:** Python 3.10+
- **GPU:** NVIDIA CUDA 12.x (recomendado para treinamento)
- **Armazenamento:** ~5 GB para datasets + modelos
- **Tempo de treinamento:** ~2-4h por modelo (baseline centralizado com GPU)

---

## 📅 Cronograma de Pesquisa

### Período 1: Fundação (4-6 semanas)
- [ ] Leitura dos papers fundamentais (referências Camadas 1-2)
- [ ] Download e validação dos datasets (Galaxy10 DECaLS + SDSS)
- [ ] Setup do ambiente e reprodução de baseline simples (CNN)
- [ ] Implementação de pipeline de pré-processamento multi-modal

### Período 2: Baseline Centralizado (6-8 semanas)
- [ ] Treinamento de modelos unimodais (ResNet, EfficientNet, ViT)
- [ ] Implementação de early fusion / late fusion architecture
- [ ] Validação que multi-modal > unimodal em configuração centralizada
- [ ] Análise de erros e Grad-CAM para modelos visuais
- [ ] SHAP analysis para contribuição de features tabulares

### Período 3: Federated Learning (8-10 semanas)
- [ ] Implementação de FedAvg e FedProx com Flower
- [ ] Criação de simulação de nós heterogêneos (4 nós com non-IID)
- [ ] Comparação: FedAvg unimodal vs. FedProx unimodal vs. FedAvg multi-modal
- [ ] Análise de convergência e comunicação
- [ ] Quantificação de melhoria de acurácia devido à multi-modalidade

### Período 4: Explicabilidade e Anomalias (6-8 semanas)
- [ ] Identificação de galáxias anômalas (low confidence + outliers)
- [ ] Aplicação de Grad-CAM + SHAP simultaneamente
- [ ] Desenvolvimento de método de decomposição de anomalias por modalidade
- [ ] Validação física com astrônomos ou Galaxy Zoo volunteers
- [ ] Documentação de insights descobertos

### Período 5: Escrita e Disseminação (4-6 semanas)
- [ ] Redação de paper científico (target venue: IEEE Transactions on Geoscience and Remote Sensing ou NeurIPS Workshop on Federated Learning)
- [ ] Preparação de code release com documentação
- [ ] Criação de demos e visualizações interativas
- [ ] Submission a conferências / journals

---

## 📈 Métricas de Sucesso

### Baseline Centralizado
- ✅ Multi-modal > Unimodal em acurácia (diferença > 2%)
- ✅ Convergência <200 epochs para modelos fine-tuned
- ✅ F1-macro ≥ 85% no test set

### Federated Learning
- ✅ FedAvg + multi-modal mitiga ≥50% da degradação de acurácia vs. FedAvg unimodal em non-IID
- ✅ Convergência em <100 rodadas de comunicação
- ✅ Comunicação total reduzida em ≥30% com compressão de gradientes

### Explicabilidade
- ✅ 70%+ de concordância entre decomposição de anomalias e feedback de especialistas
- ✅ Descoberta de ≥3 padrões físicos não-óbvios em galáxias anômalas

### Publicação
- ✅ 1+ paper publicado ou aceito em venue de tier A (IEEE Trans., NeurIPS workshop, ICML)
- ✅ Code disponível em GitHub com reprodutibilidade verificada
- ✅ ≥2 apresentações em eventos acadêmicos

---

## 🎯 Questões de Pesquisa Abertas

1. **Quanto a multi-modalidade ajuda em FL non-IID?** Existe um threshold mínimo de correlação entre modalidades que garanta estabilização?

2. **Qual mecanismo de fusão é ótimo para FL?** Late fusion (nossa proposta) vs. early fusion vs. joint fusion em heterogeneidade.

3. **XAI revela fenômenos físicos desconhecidos?** Há padrões em galáxias anômalas que indicam física não-compreendida?

4. **Generalização entre observatórios:** Um modelo federado treinado com SDSS + DECaLS generaliza para dados de LSST ou Euclid?

5. **Privacidade e compressão:** Quantos bits de comunicação são necessários para preservar benefício de MMFL mantendo privacidade diferencial?

---

## 📚 Referências Bibliográficas

### 🔵 Camada 1 — Fundamentos do Federated Learning

1. **McMahan et al. (2017)** — FedAvg original
   - "Communication-Efficient Learning of Deep Networks from Decentralized Data"
   - [arxiv.org/abs/1602.05629](https://arxiv.org/abs/1602.05629)
   - O paper fundador de FL. Define o ciclo local→servidor→global e análise inicial de problemas non-IID.

2. **Li et al. (2020)** — FedProx
   - "Federated Optimization in Heterogeneous Networks"
   - [arxiv.org/abs/1812.06127](https://arxiv.org/abs/1812.06127)
   - Introduz o termo proximal μ para estabilizar treinamento em dados non-IID. Principal baseline para comparação.

3. **Kairouz et al. (2021)** — Survey abrangente de FL
   - "Advances and Open Problems in Federated Learning"
   - [arxiv.org/abs/1912.04977](https://arxiv.org/abs/1912.04977)
   - Survey definitivo (200+ páginas). Seções 1–3 e 6 cobrem base teórica completa sobre heterogeneidade.

4. **Zhu et al. (2021)** — FL em dados non-IID
   - "Federated Learning on Non-IID Data: A Survey"
   - [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0925231221013254)
   - Classifica deterioração de acurácia e abordagens de solução (baseadas em dados vs. algoritmos).

5. **Astrini et al. (2024)** — Taxonomia atualizada de non-IID
   - "Non-IID Data in Federated Learning: A Survey with Taxonomy, Metrics, Methods, Frameworks and Future Directions"
   - [arxiv.org/abs/2411.12377](https://arxiv.org/abs/2411.12377)
   - Survey mais recente. Cobre 235 papers e mostra crescimento consistente de pesquisa em non-IID desde 2020.

### 🟣 Camada 2 — Classificação de Galáxias com Deep Learning

6. **Lintott et al. (2008)** — Galaxy Zoo original
   - "Galaxy Zoo: Morphologies derived from visual inspection of galaxies from the Sloan Digital Sky Survey"
   - [arxiv.org/abs/0804.4483](https://arxiv.org/abs/0804.4483)
   - Base de todos os labels do Galaxy Zoo. Justifica origem e confiabilidade das anotações.

7. **Walmsley et al. (2022)** — Galaxy Zoo DECaLS + Zoobot (ESTADO DA ARTE ATUAL)
   - "Galaxy Zoo DECaLS: Detailed Visual Morphology Measurements from Volunteers and Deep Learning for 314,000 Galaxies"
   - [arxiv.org/abs/2102.08414](https://arxiv.org/abs/2102.08414)
   - Classificações morfológicas para 314K galáxias DECaLS. Ensemble de redes Bayesianas como baseline centralizado.

8. **Zhao et al. (2024)** — CvT no benchmark Galaxy10 DECaLS
   - "Galaxy Morphology Classification Based on Convolutional Vision Transformer (CvT)"
   - [aanda.org/articles/aa/full_html/2024/03/aa48544-23](https://www.aanda.org/articles/aa/full_html/2024/03/aa48544-23/aa48544-23.html)
   - SOTA atual: 98.8% acurácia. Upper bound de performance para comparação centralizada.

9. **Dey et al. (2019)** — DECaLS survey
   - "Overview of the DESI Legacy Imaging Surveys"
   - [arxiv.org/abs/1804.08657](https://arxiv.org/abs/1804.08657)
   - Caracterização do survey do qual vêm as imagens do Galaxy10 DECaLS.

### 🟢 Camada 3 — Fusão Multi-modal (Imagem + Tabular)

10. **Hager et al. (2023)** — CVPR — Fusão contrastiva multi-modal
    - "Best of Both Worlds: Multimodal Contrastive Learning with Tabular and Imaging Data"
    - [arxiv.org/abs/2303.14080](https://arxiv.org/abs/2303.14080)
    - Late fusion com aprendizado contrastivo entre modalidades. Fundação teórica da arquitetura proposta.

11. **Wolf et al. (2022)** — DAFT: Fusão dinâmica em CNNs
    - "DAFT: A Universal Module to Interweave Tabular Data and 3D Images in CNNs"
    - [arxiv.org/abs/2107.12805](https://arxiv.org/abs/2107.12805)
    - Dynamic Affine Feature Map Transform para fusão de imagens e dados tabulares. Alternativa não explorada.

12. **Huang et al. (2020)** — Survey de fusão clínica
    - "Fusion of Medical Imaging and Electronic Health Records Using Deep Learning: A Systematic Review and Implementation Guidelines"
    - [arxiv.org/abs/2001.03741](https://arxiv.org/abs/2001.03741)
    - Categoriza early/joint/late fusion com exemplos práticos. Justifica escolha de late fusion.

### 🟠 Camada 4 — Federated Learning Multi-modal (Originalidade)

13. **Springer 2024** — Survey MMFL
    - "A Survey of Multimodal Federated Learning: Background, Applications, and Perspectives"
    - [link.springer.com/article/10.1007/s00530-024-01422-9](https://link.springer.com/article/10.1007/s00530-024-01422-9)
    - MMFL ainda é campo emergente. Mostra lacuna de pesquisa que o projeto preenche.

14. **Tao et al. (2024)** — FedMM em patologia
    - "FedMM: Federated Multi-Modal Learning with Modality Heterogeneity in Computational Pathology"
    - [arxiv.org/abs/2402.15858](https://arxiv.org/abs/2402.15858)
    - Trabalho mais próximo estruturalmente. Treina extratores single-modal de forma federada garantindo privacidade.

15. **FedFusion (2023)** — FL multi-modal em sensoriamento remoto
    - "FedFusion: Manifold Driven Federated Learning for Multi-Satellite and Multi-Modality Fusion"
    - IEEE Transactions on Geoscience and Remote Sensing
    - Único trabalho com domínio próximo (dados de satélite). Citar diretamente na motivação.

16. **Lin et al. (2023)** — Survey geral MMFL
    - "Federated Learning on Multimodal Data: A Comprehensive Survey"
    - [link.springer.com/article/10.1007/s11633-022-1398-0](https://link.springer.com/article/10.1007/s11633-022-1398-0)
    - Referência base para trabalhos relacionados em MMFL.

### 🔴 Camada 5 — Papers de Suporte (Framework, Particionamento)

17. **Beutel et al. (2022)** — Flower: Framework de FL
    - "Flower: A Friendly Federated Learning Research Framework"
    - [arxiv.org/abs/2007.14390](https://arxiv.org/abs/2007.14390)
    - Ferramenta de simulação para experimentos. Citar na seção de implementação.

18. **Li et al. (2022)** — Estudo empírico non-IID (ICDE)
    - "Federated Learning on Non-IID Data Silos: An Experimental Study"
    - [arxiv.org/abs/2102.02079](https://arxiv.org/abs/2102.02079)
    - Benchmark empírico: FedAvg vs. FedProx vs. SCAFFOLD em tipos diversos de non-IID. Template de experimento.

19. **Avaliação de impacto non-IID (2025)**
    - "A Thorough Assessment of the Non-IID Data Impact in Federated Learning"
    - [arxiv.org/abs/2503.17070](https://arxiv.org/abs/2503.17070)
    - Demonstra threshold crítico: quando distância Hellinger > 0.75, queda de acurácia torna-se drástica.

---

### 📚 Ordem de Leitura Sugerida

| Período | Foco | Referências |
|---------|------|-------------|
| **Semana 1** | Domínio: Galaxy Zoo | [6](#6-lintott-et-al-2008) → [7](#7-walmsley-et-al-2022) → [9](#9-dey-et-al-2019) |
| **Semana 2** | FL Clássico: FedAvg/FedProx | [1](#1-mcmahan-et-al-2017) → [2](#2-li-et-al-2020) → [4](#4-zhu-et-al-2021) |
| **Semana 3** | FL Avançado + non-IID | [3](#3-kairouz-et-al-2021) → [5](#5-astrini-et-al-2024) → [18](#18-li-et-al-2022) |
| **Semana 4** | Fusão Multi-modal | [10](#10-hager-et-al-2023) → [11](#11-wolf-et-al-2022) → [12](#12-huang-et-al-2020) |
| **Semana 5** | MMFL: Estado da Arte | [13](#13-springer-2024) → [14](#14-tao-et-al-2024) → [16](#16-lin-et-al-2023) |
| **Semana 6** | Suporte + Validação | [15](#15-fedfusion-2023) → [17](#17-beutel-et-al-2022) → [19](#19-avaliação-de-impacto-non-iid-2025) → [8](#8-zhao-et-al-2024) |
