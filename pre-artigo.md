# Pre-artigo: classificacao morfologica de galaxias com avaliacao robusta

## 1. Resumo executivo

Este documento reune os principais resultados em `docs/` para orientar a escrita do artigo. A recomendacao e seguir com a escrita agora, usando os resultados atuais como base. Ha cobertura suficiente para sustentar uma narrativa cientifica: foram registradas 36 runs, 21 pares unicos de modelo/dataset no leaderboard geral e 14 pares naturais no leaderboard robusto.

A leitura principal deve vir do `docs/leaderboard_robust.md`, porque ele filtra os resultados para avaliacoes em datasets naturais. O `docs/leaderboard.md` deve ser usado como apoio para discutir o efeito de balanceamento artificial, especialmente SMOTE e random oversampling, mas nao como evidencia principal de generalizacao.

Tese sugerida para o artigo:

- O balanceamento melhora muito o desempenho observado, mas pode superestimar a performance pratica quando avaliado em dados processados.
- Em dados naturais, DECals apresenta resultados mais estaveis que SDSS.
- SDSS raw sofre forte efeito de desbalanceamento: a acuracia fica alta, mas a balanced accuracy cai.
- O Federated DINO e competitivo em cenarios intra-dominio, mas a transferencia SDSS <-> DECals ainda e fraca.
- Balanced accuracy, macro F1, kappa e log loss devem ser priorizadas em relacao a acuracia simples.

## 2. Fontes consolidadas

Arquivos principais:

- `docs/leaderboard_robust.md`: ranking principal para o artigo, com 14 pares naturais.
- `docs/leaderboard.md`: ranking geral com 21 pares, incluindo datasets processados.
- `docs/recommendations.md`: diagnosticos automaticos de desbalanceamento, classes nao recuperadas e confusoes sistematicas.
- `docs/by_dataset/*/comparison.md`: comparacoes por dataset.
- `docs/by_model/*/comparison.md`: comparacoes por modelo.
- `docs/dataset/*/summary.md`: descricao visual e estatistica dos datasets.
- `docs/xai/*`: amostras e explicacoes visuais para analise qualitativa.

Figuras candidatas:

- `docs/leaderboard_balanced_accuracy.png`
- `docs/leaderboard_macro_f1.png`
- `docs/leaderboard_log_loss.png`
- `docs/by_dataset/sdss_raw/comparison.png`
- `docs/by_dataset/decals_raw/comparison.png`
- `docs/by_model/federated_dino/comparison.png`
- mosaicos de amostras em `docs/dataset/*/sample_mosaic.png`

## 3. Resultados principais em avaliacao natural

Tabela recomendada para a secao de resultados. Estes valores devem ser tratados como o nucleo quantitativo do artigo.

| Rank | Modelo | Dataset/cenario | Accuracy | Balanced accuracy | Macro F1 | Kappa | Log loss | Erros |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | federated_dino | client_decals_raw_on_decals_raw | 0.8213 | 0.8299 | 0.6953 | 0.7881 | 0.7632 | 342 |
| 2 | federated_dino | global_on_sdss_raw | 0.8476 | 0.8146 | 0.8026 | 0.8021 | 0.4718 | 498 |
| 3 | dino | decals_raw | 0.8129 | 0.8049 | 0.7958 | 0.7887 | 0.7301 | 498 |
| 4 | federated_dino | client_sdss_raw_on_sdss_raw | 0.8387 | 0.7872 | 0.7876 | 0.7900 | 0.5042 | 527 |
| 5 | resnet50 | decals_raw | 0.8023 | 0.7827 | 0.7800 | 0.7771 | 0.6444 | 526 |
| 6 | efficientnet | decals_raw | 0.7971 | 0.7810 | 0.7763 | 0.7706 | 0.6284 | 540 |
| 7 | efficientnet | sdss_raw | 0.8341 | 0.6985 | 0.6912 | 0.7863 | 0.4874 | 542 |
| 8 | resnet50 | sdss_raw | 0.8427 | 0.6895 | 0.6941 | 0.7959 | 0.5393 | 514 |
| 9 | dino | sdss_raw | 0.8421 | 0.6829 | 0.6834 | 0.7957 | 0.6076 | 516 |
| 10 | federated_dino | global_on_decals_raw | 0.6458 | 0.6243 | 0.5410 | 0.5844 | 1.0597 | 678 |
| 11 | federated_dino | client_decals_raw_on_sdss_raw | 0.5685 | 0.5383 | 0.4679 | 0.4645 | 1.4319 | 1410 |
| 12 | dino | train_sdss_raw_eval_decals_raw | 0.5308 | 0.4983 | 0.4352 | 0.4468 | 1.5839 | 898 |
| 13 | federated_dino | client_sdss_raw_on_decals_raw | 0.5470 | 0.4837 | 0.4445 | 0.4634 | 1.4915 | 867 |
| 14 | dino | train_decals_raw_eval_sdss_raw | 0.4755 | 0.4377 | 0.3756 | 0.3040 | 2.0288 | 1714 |

Interpretacao direta:

- O melhor resultado natural e `federated_dino` em `client_decals_raw_on_decals_raw`, com balanced accuracy de 0.8299.
- O melhor resultado global em SDSS natural e `federated_dino` em `global_on_sdss_raw`, com balanced accuracy de 0.8146.
- Entre os modelos centralizados em DECals raw, `dino` lidera com balanced accuracy de 0.8049.
- Em SDSS raw, `efficientnet` fica ligeiramente acima dos demais em balanced accuracy, mas a diferenca para `resnet50` e `dino` e pequena.
- Os cenarios cross-dataset sao os mais fracos: `train_sdss_raw_eval_decals_raw` fica em 0.4983 e `train_decals_raw_eval_sdss_raw` em 0.4377 de balanced accuracy.

## 4. Resultados com datasets processados

Estes resultados sao importantes para discutir o efeito do balanceamento, mas devem aparecer separados dos resultados naturais.

| Rank geral | Modelo | Dataset processado | Accuracy | Balanced accuracy | Macro F1 | Kappa | Log loss | Erros |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | dino | sdss_random_over_sampling | 0.9764 | 0.9764 | 0.9763 | 0.9737 | 0.1162 | 248 |
| 2 | efficientnet | sdss_smote | 0.9721 | 0.9721 | 0.9720 | 0.9690 | 0.1229 | 293 |
| 3 | resnet50 | sdss_smote | 0.9710 | 0.9710 | 0.9708 | 0.9678 | 0.1201 | 304 |
| 4 | dino | sdss_smote | 0.9643 | 0.9643 | 0.9640 | 0.9603 | 0.1814 | 375 |
| 5 | dino | decals_smote | 0.8760 | 0.8760 | 0.8759 | 0.8622 | 0.4759 | 492 |
| 6 | resnet50 | decals_smote | 0.8667 | 0.8666 | 0.8655 | 0.8519 | 0.4432 | 529 |
| 7 | efficientnet | decals_smote | 0.8561 | 0.8560 | 0.8544 | 0.8401 | 0.4439 | 571 |

Interpretacao sugerida:

- SDSS processado atinge resultados muito altos, chegando a 0.9764 de balanced accuracy com DINO e random oversampling.
- Esse ganho deve ser apresentado como evidencia de que o desbalanceamento era um fator dominante.
- A comparacao mais justa para generalizacao permanece a avaliacao natural, especialmente porque o objetivo do artigo e classificar galaxias em condicoes realistas.
- DECals com SMOTE melhora, mas nao chega ao mesmo patamar de SDSS processado, sugerindo maior dificuldade visual ou maior variabilidade do dominio.

## 5. Achados para discussao

### 5.1 SDSS raw e dominancia por desbalanceamento

Em SDSS raw, a diferenca entre accuracy e balanced accuracy e grande. Exemplos:

- `efficientnet/sdss_raw`: accuracy 0.8341 contra balanced accuracy 0.6985.
- `resnet50/sdss_raw`: accuracy 0.8427 contra balanced accuracy 0.6895.
- `dino/sdss_raw`: accuracy 0.8421 contra balanced accuracy 0.6829.

Isso indica que a acuracia simples e otimista. O modelo acerta muitas amostras das classes majoritarias, mas nao recupera bem todas as classes. O artigo deve justificar a escolha de balanced accuracy e macro F1 como metricas principais.

### 5.2 Classe minoritaria critica em SDSS

O `docs/recommendations.md` aponta repetidamente que `class_5` em SDSS raw nao e recuperada por diferentes modelos, com recall/per-class accuracy igual a 0.000 em varias runs. Isso e um resultado importante: o problema nao e apenas arquitetura, mas tambem distribuicao dos dados.

Uso no artigo:

- mencionar como limitacao experimental;
- mostrar que balanceamento no treino e necessario;
- evitar conclusoes baseadas apenas em accuracy;
- sugerir focal loss, oversampling de treino ou coleta de mais amostras como trabalho futuro.

### 5.3 DECals raw e mais estavel que SDSS raw

Em DECals raw, os modelos centralizados ficam proximos:

- `dino`: balanced accuracy 0.8049.
- `resnet50`: balanced accuracy 0.7827.
- `efficientnet`: balanced accuracy 0.7810.

Isso sugere que DECals oferece um cenario natural mais equilibrado para comparacao entre arquiteturas. O DINO lidera, mas a vantagem sobre ResNet50 e EfficientNet nao e grande o bastante para afirmar superioridade absoluta sem discutir variancia e repeticoes.

### 5.4 Federated DINO

O Federated DINO tem o melhor resultado natural no cenario `client_decals_raw_on_decals_raw`, com balanced accuracy de 0.8299. Tambem vai bem em `global_on_sdss_raw`, com 0.8146.

Ao mesmo tempo, o desempenho cai em `global_on_decals_raw` para 0.6243 e nos cenarios cruzados:

- `client_decals_raw_on_sdss_raw`: 0.5383.
- `client_sdss_raw_on_decals_raw`: 0.4837.

Interpretacao recomendada:

- o aprendizado federado e promissor quando treino e avaliacao permanecem proximos do dominio local;
- a federacao nao elimina automaticamente o problema de domain shift;
- a diferenca entre SDSS e DECals ainda precisa de estrategias explicitas de adaptacao de dominio.

### 5.5 Transferencia cruzada SDSS <-> DECals

Os cenarios `train_sdss_raw_eval_decals_raw` e `train_decals_raw_eval_sdss_raw` sao os resultados mais fracos entre os experimentos naturais centralizados:

- SDSS -> DECals: balanced accuracy 0.4983.
- DECals -> SDSS: balanced accuracy 0.4377.

Este e um dos resultados mais relevantes do artigo. Ele mostra que bom desempenho intra-dataset nao implica generalizacao cross-dataset. A discussao deve conectar isso a mudancas de dominio, diferencas instrumentais, distribuicao de classes e variabilidade visual.

## 6. Estrutura sugerida para o artigo

### Introducao

Problema: classificacao morfologica de galaxias e sensivel a desbalanceamento, diferencas de dominio e escolha de metrica. Apresentar SDSS e DECals como fontes com caracteristicas distintas, e justificar a comparacao entre arquiteturas CNN, DINO e Federated DINO.

Contribuicoes sugeridas:

- avaliacao comparativa de DINO, EfficientNet, ResNet50 e Federated DINO em datasets de galaxias;
- analise do impacto de balanceamento em SDSS e DECals;
- avaliacao natural e cross-dataset para medir generalizacao;
- diagnostico de classes problematicas e confusoes sistematicas;
- uso de XAI como apoio qualitativo.

### Trabalhos relacionados

Organizar em tres blocos:

- classificacao morfologica de galaxias com deep learning;
- aprendizado auto-supervisionado/transformers visuais em astronomia ou visao computacional;
- aprendizado federado e generalizacao entre dominios.

### Metodologia

Descrever:

- datasets SDSS e DECals;
- versoes raw e processadas;
- SMOTE e random oversampling;
- modelos DINO, EfficientNet, ResNet50 e Federated DINO;
- protocolo de treino, validacao e teste;
- diferenca entre avaliacao processada e avaliacao natural;
- metricas: balanced accuracy como metrica principal, accuracy como secundaria, macro F1, kappa, MCC, ROC AUC macro e log loss.

### Resultados e discussao

Ordem recomendada:

1. Resultados naturais do leaderboard robusto.
2. Comparacao SDSS raw vs DECals raw.
3. Impacto do balanceamento artificial.
4. Resultados federados.
5. Cross-dataset e domain shift.
6. Analise de erros e classes problematicas.
7. XAI como evidencia qualitativa.

### Conclusao

Mensagem central:

- balanceamento e avaliacao por metrica macro sao essenciais;
- resultados processados mostram potencial, mas resultados naturais sao mais honestos;
- DINO e Federated DINO sao competitivos, mas a transferencia entre SDSS e DECals segue sendo uma limitacao;
- trabalhos futuros devem focar em adaptacao de dominio, mais repeticoes por seed, focal loss/oversampling de treino e analise visual das classes confundidas.

## 7. Limitacoes a declarar

- Algumas combinacoes modelo/dataset possuem apenas uma run, entao nao ha intervalo de confianca ou desvio padrao.
- Varias runs registram `git_is_dirty: true`, o que exige cuidado ao discutir reprodutibilidade exata.
- Em algumas runs federadas/cross-dataset, `roc_auc_macro` aparece como `n/a`.
- Os resultados processados nao devem ser comparados diretamente com resultados naturais como se tivessem o mesmo protocolo.
- Confusoes sistematicas entre classes podem refletir ambiguidade real de rotulo, nao apenas erro do modelo.

## 8. Experimentos opcionais antes da versao final

Nao sao bloqueadores para escrever o artigo. Se houver tempo computacional, estes sao os experimentos que mais agregariam:

- repetir os principais baselines naturais com 3 seeds para estimar variancia;
- testar focal loss ou class-weighted loss em SDSS raw;
- rodar oversampling apenas no treino para atacar `class_5`, mantendo validacao e teste naturais;
- revisar amostras de pares de classes com confusao sistematica em `docs/recommendations.md`;
- gerar uma selecao pequena de figuras XAI para os melhores modelos em SDSS raw e DECals raw.

## 9. Decisao recomendada

O caminho recomendado e considerar os resultados atuais completos para a etapa de escrita e montar o artigo a partir deste pre-artigo. Rodar mais experimentos agora so vale se houver uma pergunta especifica surgida durante a escrita, nao como requisito para iniciar o texto.

Prioridade imediata:

1. Transformar este documento em texto academico no `artigo.tex`.
2. Escolher 2 ou 3 figuras principais.
3. Preparar uma tabela compacta com os resultados naturais.
4. Usar os resultados processados apenas em uma subseccao sobre balanceamento.
5. Encerrar com uma discussao honesta sobre desbalanceamento, domain shift e limites de generalizacao.
