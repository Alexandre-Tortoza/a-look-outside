# A look outside

Este projeto tem como objetivo comparar diferentes arquiteturas de modelos de aprendizado profundo aplicadas à classificação de galáxias, utilizando múltiplos datasets astronômicos e avaliando sua capacidade de generalização entre domínios distintos.

---

## Objetivos

- Comparar diferentes arquiteturas:
  - CNNs tradicionais
  - Vision Transformers
  - Modelos leves (MobileNet, EasyNet)

- Avaliar desempenho intra-dataset e cross-dataset
- Investigar técnicas para tratamento de desbalanceamento
- Melhorar desempenho em classes com menor acurácia

---

## Modelos Avaliados

| Modelo             | Descrição                                                   |
| ------------------ | ----------------------------------------------------------- |
| CNN                | Arquitetura convolucional clássica para visão computacional |
| Vision Transformer | Modelo baseado em mecanismos de atenção                     |
| MobileNet          | Modelo leve otimizado para eficiência                       |
| EasyNet            | Arquitetura leve com baixo custo computacional              |

---

## Datasets

| Dataset | Descrição                        |
| ------- | -------------------------------- |
| SDSS    | Sloan Digital Sky Survey         |
| DECaLS  | Dark Energy Camera Legacy Survey |

---

## Estratégia Experimental

### 1. Treinamento Base

Treinar e avaliar modelos em seus respectivos datasets:

| Treino | Teste  |
| ------ | ------ |
| SDSS   | SDSS   |
| DECaLS | DECaLS |

---

### 2. Avaliação Cross-Dataset

Avaliar capacidade de generalização:

| Treino | Teste  |
| ------ | ------ |
| SDSS   | DECaLS |
| DECaLS | SDSS   |

---

### 3. Dataset Combinado

- Criar um dataset unificado: SDSS + DECaLS

Avaliações:

| Modelo Treinado em | Testado em Dataset Combinado |
| ------------------ | ---------------------------- |
| SDSS               | SDSS + DECaLS                |
| DECaLS             | SDSS + DECaLS                |

---

### 4. Treinamento Unificado

Treinar um novo modelo com o dataset combinado:

| Treino        | Teste  |
| ------------- | ------ |
| SDSS + DECaLS | SDSS   |
| SDSS + DECaLS | DECaLS |

---

## Tratamento de Desbalanceamento

| Técnica             | Descrição                                   |
| ------------------- | ------------------------------------------- |
| Stratified Sampling | Mantém proporção de classes nos conjuntos   |
| Data Augmentation   | Transformações como rotação, flip, zoom     |
| Oversampling        | Aumenta amostras de classes minoritárias    |
| Undersampling       | Reduz amostras de classes majoritárias      |
| Class Weights       | Penaliza mais erros em classes minoritárias |

---

## Avaliação de Desempenho

### Métricas

| Métrica   | Descrição                                                    |
| --------- | ------------------------------------------------------------ |
| Accuracy  | Proporção total de acertos                                   |
| Precision | Proporção de verdadeiros positivos entre positivos previstos |
| Recall    | Proporção de verdadeiros positivos encontrados               |
| F1-Score  | Média harmônica entre precision e recall                     |

---

### Matriz de Confusão

A matriz de confusão permite analisar:

- Distribuição de erros por classe
- Classes mais confundidas
- Desempenho individual por categoria

---

### Métricas por Tipo de Modelo

| Modelo        | Métricas / Análises                                    |
| ------------- | ------------------------------------------------------ |
| CNN           | Accuracy, Loss, curvas de aprendizado                  |
| Transformer   | Attention maps, análise de patches, interpretabilidade |
| Modelos leves | Trade-off entre acurácia e custo computacional         |

---

## Cookbook de Variáveis e Hiperparâmetros

### Hiperparâmetros

| Parâmetro     | Descrição                                 |
| ------------- | ----------------------------------------- |
| Learning Rate | Taxa de atualização dos pesos             |
| Batch Size    | Número de amostras por batch              |
| Epochs        | Número de iterações completas no dataset  |
| Optimizer     | Algoritmo de otimização (Adam, SGD, etc.) |
| Loss Function | Função de erro (ex: Cross-Entropy)        |

---

### Variáveis Experimentais

| Variável                 | Descrição                 |
| ------------------------ | ------------------------- |
| Arquitetura do modelo    | CNN, Transformer, Mobile  |
| Dataset de treino        | SDSS, DECaLS, combinado   |
| Dataset de teste         | SDSS, DECaLS, combinado   |
| Técnica de balanceamento | Método aplicado           |
| Data augmentation        | Transformações utilizadas |

---

## Próximos Passos

- Identificar classes com pior desempenho
- Aplicar técnicas específicas:
  - Fine-tuning direcionado
  - Rebalanceamento localizado
  - Loss functions customizadas (ex: focal loss)

- Melhorar generalização entre datasets

---

## Conclusão

O foco do projeto é avaliar não apenas o desempenho isolado dos modelos, mas sua robustez e capacidade de generalização entre diferentes domínios astronômicos, utilizando estratégias experimentais progressivas e análise detalhada de erros.

---

Se quiser, posso adaptar isso para formato de artigo (IEEE/ACM) ou incluir estrutura de código (pastas, scripts, pipeline).
