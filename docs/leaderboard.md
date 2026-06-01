# Leaderboard — atualizado 2026-05-31 23:52 UTC

- Primary metric: **balanced_accuracy**
- Secondary metric: **accuracy**
- Total runs registered: **36**
- Unique (model, dataset) pairs: **21**

## Top results (sorted by balanced_accuracy)

|   rank | model_name     | dataset_name                    | evaluation_dataset_kind   | robust_evaluation   |   accuracy |   balanced_accuracy |   macro_f1 |   cohen_kappa |   matthews_correlation_coefficient | roc_auc_macro   |   log_loss |   misclassified_count |   duration_seconds | documentation_directory                                                                                                        |
|-------:|:---------------|:--------------------------------|:--------------------------|:--------------------|-----------:|--------------------:|-----------:|--------------:|-----------------------------------:|:----------------|-----------:|----------------------:|-------------------:|:-------------------------------------------------------------------------------------------------------------------------------|
|      1 | dino           | sdss_random_over_sampling       | processed                 | False               |     0.9764 |              0.9764 |     0.9763 |        0.9737 |                             0.9738 | 0.9985          |     0.1162 |                   248 |               5720 | /home/alexmrtr/Pucpr/a-look-outside/docs/models/dino/dino-sdss_random_over_sampling-10-05-2026-2-sdss_random_over_sampling     |
|      2 | efficientnet   | sdss_smote                      | processed                 | False               |     0.9721 |              0.9721 |     0.972  |        0.969  |                             0.969  | 0.9987          |     0.1229 |                   293 |               2870 | /home/alexmrtr/a-look-outside/docs/models/efficientnet/efficientnet-sdss_smote-26-05-2026-sdss_smote                           |
|      3 | resnet50       | sdss_smote                      | processed                 | False               |     0.971  |              0.971  |     0.9708 |        0.9678 |                             0.9679 | 0.9985          |     0.1201 |                   304 |               3694 | /home/alexmrtr/a-look-outside/docs/models/resnet50/resnet50-sdss_smote-26-05-2026-sdss_smote                                   |
|      4 | dino           | sdss_smote                      | processed                 | False               |     0.9643 |              0.9643 |     0.964  |        0.9603 |                             0.9603 | 0.9935          |     0.1814 |                   375 |               5346 | /home/alexmrtr/a-look-outside/docs/models/dino/dino-sdss_smote-31-05-2026-sdss_smote                                           |
|      5 | dino           | decals_smote                    | processed                 | False               |     0.876  |              0.876  |     0.8759 |        0.8622 |                             0.8624 | 0.9820          |     0.4759 |                   492 |               1661 | /home/alexmrtr/a-look-outside/docs/models/dino/dino-decals_smote-26-05-2026-decals_smote                                       |
|      6 | resnet50       | decals_smote                    | processed                 | False               |     0.8667 |              0.8666 |     0.8655 |        0.8519 |                             0.8521 | 0.9861          |     0.4432 |                   529 |                965 | /home/alexmrtr/a-look-outside/docs/models/resnet50/resnet50-decals_smote-26-05-2026-decals_smote                               |
|      7 | efficientnet   | decals_smote                    | processed                 | False               |     0.8561 |              0.856  |     0.8544 |        0.8401 |                             0.8404 | 0.9851          |     0.4439 |                   571 |                603 | /home/alexmrtr/a-look-outside/docs/models/efficientnet/efficientnet-decals_smote-26-05-2026-decals_smote                       |
|      8 | federated_dino | client_decals_raw_on_decals_raw | natural                   | True                |     0.8213 |              0.8299 |     0.6953 |        0.7881 |                             0.7927 | n/a             |     0.7632 |                   342 |                696 | /home/alexmrtr/a-look-outside/docs/models/federated_dino/federated_dino-sdss_decals-31-05-2026-client_decals_raw_on_decals_raw |
|      9 | federated_dino | global_on_sdss_raw              | natural                   | True                |     0.8476 |              0.8146 |     0.8026 |        0.8021 |                             0.8044 | 0.9788          |     0.4718 |                   498 |                673 | /home/alexmrtr/a-look-outside/docs/models/federated_dino/federated_dino-sdss_decals-31-05-2026-global_on_sdss_raw              |
|     10 | dino           | decals_raw                      | natural                   | True                |     0.8129 |              0.8049 |     0.7958 |        0.7887 |                             0.789  | 0.9552          |     0.7301 |                   498 |               1392 | /home/alexmrtr/a-look-outside/docs/models/dino/dino-decals_raw-31-05-2026-decals_raw                                           |
|     11 | federated_dino | client_sdss_raw_on_sdss_raw     | natural                   | True                |     0.8387 |              0.7872 |     0.7876 |        0.79   |                             0.7923 | 0.9719          |     0.5042 |                   527 |                683 | /home/alexmrtr/a-look-outside/docs/models/federated_dino/federated_dino-sdss_decals-31-05-2026-client_sdss_raw_on_sdss_raw     |
|     12 | resnet50       | decals_raw                      | natural                   | True                |     0.8023 |              0.7827 |     0.78   |        0.7771 |                             0.7779 | 0.9690          |     0.6444 |                   526 |                659 | /home/alexmrtr/a-look-outside/docs/models/resnet50/resnet50-decals_raw-26-05-2026-decals_raw                                   |
|     13 | efficientnet   | decals_raw                      | natural                   | True                |     0.7971 |              0.781  |     0.7763 |        0.7706 |                             0.7712 | 0.9691          |     0.6284 |                   540 |                418 | /home/alexmrtr/a-look-outside/docs/models/efficientnet/efficientnet-decals_raw-26-05-2026-decals_raw                           |
|     14 | efficientnet   | sdss_raw                        | natural                   | True                |     0.8341 |              0.6985 |     0.6912 |        0.7863 |                             0.787  | 0.9657          |     0.4874 |                   542 |               1152 | /home/alexmrtr/a-look-outside/docs/models/efficientnet/efficientnet-sdss_raw-31-05-2026-sdss_raw                               |
|     15 | resnet50       | sdss_raw                        | natural                   | True                |     0.8427 |              0.6895 |     0.6941 |        0.7959 |                             0.7973 | 0.9425          |     0.5393 |                   514 |               2120 | /home/alexmrtr/a-look-outside/docs/models/resnet50/resnet50-sdss_raw-31-05-2026-sdss_raw                                       |
|     16 | dino           | sdss_raw                        | natural                   | True                |     0.8421 |              0.6829 |     0.6834 |        0.7957 |                             0.7972 | 0.9086          |     0.6076 |                   516 |               1270 | /home/alexmrtr/a-look-outside/docs/models/dino/dino-sdss_raw-26-05-2026-sdss_raw                                               |
|     17 | federated_dino | global_on_decals_raw            | natural                   | True                |     0.6458 |              0.6243 |     0.541  |        0.5844 |                             0.592  | n/a             |     1.0597 |                   678 |                676 | /home/alexmrtr/a-look-outside/docs/models/federated_dino/federated_dino-sdss_decals-31-05-2026-global_on_decals_raw            |
|     18 | federated_dino | client_decals_raw_on_sdss_raw   | natural                   | True                |     0.5685 |              0.5383 |     0.4679 |        0.4645 |                             0.4849 | 0.9048          |     1.4319 |                  1410 |                692 | /home/alexmrtr/a-look-outside/docs/models/federated_dino/federated_dino-sdss_decals-31-05-2026-client_decals_raw_on_sdss_raw   |
|     19 | dino           | train_sdss_raw_eval_decals_raw  | natural                   | True                |     0.5308 |              0.4983 |     0.4352 |        0.4468 |                             0.4564 | n/a             |     1.5839 |                   898 |               2434 | /home/alexmrtr/a-look-outside/docs/models/dino/dino-train_sdss_raw_eval_decals_raw-31-05-2026-train_sdss_raw_eval_decals_raw   |
|     20 | federated_dino | client_sdss_raw_on_decals_raw   | natural                   | True                |     0.547  |              0.4837 |     0.4445 |        0.4634 |                             0.4686 | n/a             |     1.4915 |                   867 |                686 | /home/alexmrtr/a-look-outside/docs/models/federated_dino/federated_dino-sdss_decals-31-05-2026-client_sdss_raw_on_decals_raw   |
|     21 | dino           | train_decals_raw_eval_sdss_raw  | natural                   | True                |     0.4755 |              0.4377 |     0.3756 |        0.304  |                             0.3406 | 0.7983          |     2.0288 |                  1714 |                918 | /home/alexmrtr/a-look-outside/docs/models/dino/dino-train_decals_raw_eval_sdss_raw-31-05-2026-train_decals_raw_eval_sdss_raw   |

## Best model per dataset (by balanced_accuracy)

| dataset_name                    | best_model     |   balanced_accuracy | runner_up   | gap    |
|:--------------------------------|:---------------|--------------------:|:------------|:-------|
| client_decals_raw_on_decals_raw | federated_dino |              0.8299 |             | n/a    |
| client_decals_raw_on_sdss_raw   | federated_dino |              0.5383 |             | n/a    |
| client_sdss_raw_on_decals_raw   | federated_dino |              0.4837 |             | n/a    |
| client_sdss_raw_on_sdss_raw     | federated_dino |              0.7872 |             | n/a    |
| decals_raw                      | dino           |              0.8049 | resnet50    | 0.0222 |
| decals_smote                    | dino           |              0.876  | resnet50    | 0.0093 |
| global_on_decals_raw            | federated_dino |              0.6243 |             | n/a    |
| global_on_sdss_raw              | federated_dino |              0.8146 |             | n/a    |
| sdss_random_over_sampling       | dino           |              0.9764 |             | n/a    |
| sdss_raw                        | efficientnet   |              0.6985 | resnet50    | 0.0090 |
| sdss_smote                      | efficientnet   |              0.9721 | resnet50    | 0.0010 |
| train_decals_raw_eval_sdss_raw  | dino           |              0.4377 |             | n/a    |
| train_sdss_raw_eval_decals_raw  | dino           |              0.4983 |             | n/a    |

## Best dataset per model (by balanced_accuracy)

| model_name     | best_dataset                    |   balanced_accuracy |
|:---------------|:--------------------------------|--------------------:|
| dino           | sdss_random_over_sampling       |              0.9764 |
| efficientnet   | sdss_smote                      |              0.9721 |
| federated_dino | client_decals_raw_on_decals_raw |              0.8299 |
| resnet50       | sdss_smote                      |              0.971  |

## Model rankings by metric

### accuracy

|   rank | model_name   | dataset_name              |   accuracy |
|-------:|:-------------|:--------------------------|-----------:|
|      1 | dino         | sdss_random_over_sampling |     0.9764 |
|      2 | efficientnet | sdss_smote                |     0.9721 |
|      3 | resnet50     | sdss_smote                |     0.971  |
|      4 | dino         | sdss_smote                |     0.9643 |
|      5 | dino         | decals_smote              |     0.876  |

### balanced_accuracy

|   rank | model_name   | dataset_name              |   balanced_accuracy |
|-------:|:-------------|:--------------------------|--------------------:|
|      1 | dino         | sdss_random_over_sampling |              0.9764 |
|      2 | efficientnet | sdss_smote                |              0.9721 |
|      3 | resnet50     | sdss_smote                |              0.971  |
|      4 | dino         | sdss_smote                |              0.9643 |
|      5 | dino         | decals_smote              |              0.876  |

### macro_f1

|   rank | model_name   | dataset_name              |   macro_f1 |
|-------:|:-------------|:--------------------------|-----------:|
|      1 | dino         | sdss_random_over_sampling |     0.9763 |
|      2 | efficientnet | sdss_smote                |     0.972  |
|      3 | resnet50     | sdss_smote                |     0.9708 |
|      4 | dino         | sdss_smote                |     0.964  |
|      5 | dino         | decals_smote              |     0.8759 |

### cohen_kappa

|   rank | model_name   | dataset_name              |   cohen_kappa |
|-------:|:-------------|:--------------------------|--------------:|
|      1 | dino         | sdss_random_over_sampling |        0.9737 |
|      2 | efficientnet | sdss_smote                |        0.969  |
|      3 | resnet50     | sdss_smote                |        0.9678 |
|      4 | dino         | sdss_smote                |        0.9603 |
|      5 | dino         | decals_smote              |        0.8622 |

### matthews_correlation_coefficient

|   rank | model_name   | dataset_name              |   matthews_correlation_coefficient |
|-------:|:-------------|:--------------------------|-----------------------------------:|
|      1 | dino         | sdss_random_over_sampling |                             0.9738 |
|      2 | efficientnet | sdss_smote                |                             0.969  |
|      3 | resnet50     | sdss_smote                |                             0.9679 |
|      4 | dino         | sdss_smote                |                             0.9603 |
|      5 | dino         | decals_smote              |                             0.8624 |

### roc_auc_macro

|   rank | model_name   | dataset_name              |   roc_auc_macro |
|-------:|:-------------|:--------------------------|----------------:|
|      1 | efficientnet | sdss_smote                |          0.9987 |
|      2 | resnet50     | sdss_smote                |          0.9985 |
|      3 | dino         | sdss_random_over_sampling |          0.9985 |
|      4 | dino         | sdss_smote                |          0.9935 |
|      5 | resnet50     | decals_smote              |          0.9861 |

### log_loss

|   rank | model_name   | dataset_name              |   log_loss |
|-------:|:-------------|:--------------------------|-----------:|
|      1 | dino         | sdss_random_over_sampling |     0.1162 |
|      2 | resnet50     | sdss_smote                |     0.1201 |
|      3 | efficientnet | sdss_smote                |     0.1229 |
|      4 | dino         | sdss_smote                |     0.1814 |
|      5 | resnet50     | decals_smote              |     0.4432 |


## Recent runs (chronological, last 20)

| completed_at              | model_name     | dataset_name                    |   accuracy |   balanced_accuracy |   duration_seconds |
|:--------------------------|:---------------|:--------------------------------|-----------:|--------------------:|-------------------:|
| 2026-05-31T23:52:49+00:00 | federated_dino | client_decals_raw_on_decals_raw |     0.8213 |              0.8299 |                696 |
| 2026-05-31T23:52:45+00:00 | federated_dino | client_decals_raw_on_sdss_raw   |     0.5685 |              0.5383 |                692 |
| 2026-05-31T23:52:39+00:00 | federated_dino | client_sdss_raw_on_decals_raw   |     0.547  |              0.4837 |                686 |
| 2026-05-31T23:52:36+00:00 | federated_dino | client_sdss_raw_on_sdss_raw     |     0.8387 |              0.7872 |                683 |
| 2026-05-31T23:52:29+00:00 | federated_dino | global_on_decals_raw            |     0.6458 |              0.6243 |                676 |
| 2026-05-31T23:52:26+00:00 | federated_dino | global_on_sdss_raw              |     0.8476 |              0.8146 |                673 |
| 2026-05-31T23:40:42+00:00 | dino           | train_decals_raw_eval_sdss_raw  |     0.4755 |              0.4377 |                918 |
| 2026-05-31T23:24:47+00:00 | dino           | train_sdss_raw_eval_decals_raw  |     0.5308 |              0.4983 |               2434 |
| 2026-05-31T22:30:25+00:00 | dino           | decals_smote                    |     0.8742 |              0.8742 |               2112 |
| 2026-05-31T21:54:25+00:00 | dino           | decals_raw                      |     0.7971 |              0.7859 |               1496 |
| 2026-05-31T21:28:51+00:00 | dino           | sdss_smote                      |     0.9643 |              0.9643 |               5346 |
| 2026-05-31T19:59:36+00:00 | dino           | sdss_raw                        |     0.8204 |              0.6642 |               1960 |
| 2026-05-31T19:15:34+00:00 | dino           | decals_raw                      |     0.8129 |              0.8049 |               1392 |
| 2026-05-31T18:51:53+00:00 | dino           | sdss_raw                        |     0.8207 |              0.657  |               1989 |
| 2026-05-31T17:12:19+00:00 | resnet50       | decals_raw                      |     0.7775 |              0.7737 |                828 |
| 2026-05-31T16:58:02+00:00 | dino           | sdss_raw                        |     0.8048 |              0.6214 |               3058 |
| 2026-05-31T16:07:04+00:00 | efficientnet   | sdss_raw                        |     0.8341 |              0.6985 |               1152 |
| 2026-05-31T15:47:52+00:00 | resnet50       | sdss_raw                        |     0.8427 |              0.6895 |               2120 |
| 2026-05-26T17:14:03+00:00 | dino           | train_decals_raw_eval_sdss_raw  |     0.4899 |              0.4369 |                654 |
| 2026-05-26T17:02:38+00:00 | dino           | train_sdss_raw_eval_decals_raw  |     0.5392 |              0.4656 |               1149 |