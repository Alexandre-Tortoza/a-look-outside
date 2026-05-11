# Leaderboard — atualizado 2026-05-11 03:13 UTC

- Primary metric: **balanced_accuracy**
- Secondary metric: **accuracy**
- Total runs registered: **4**
- Unique (model, dataset) pairs: **4**

## Top results (sorted by balanced_accuracy)

|   rank | model_name   | dataset_name              |   accuracy |   balanced_accuracy |   macro_f1 |   cohen_kappa |   matthews_correlation_coefficient |   roc_auc_macro |   log_loss |   misclassified_count |   duration_seconds | documentation_directory                                                                                                    |
|-------:|:-------------|:--------------------------|-----------:|--------------------:|-----------:|--------------:|-----------------------------------:|----------------:|-----------:|----------------------:|-------------------:|:---------------------------------------------------------------------------------------------------------------------------|
|      1 | dino         | sdss_random_over_sampling |     0.9764 |              0.9764 |     0.9763 |        0.9737 |                             0.9738 |          0.9985 |     0.1162 |                   248 |               5720 | /home/alexmrtr/Pucpr/a-look-outside/docs/models/dino/dino-sdss_random_over_sampling-10-05-2026-2-sdss_random_over_sampling |
|      2 | dino         | sdss_smote                |     0.9376 |              0.9376 |     0.9378 |        0.9307 |                             0.9308 |          0.9962 |     0.2206 |                   655 |               6382 | /home/alexmrtr/Pucpr/a-look-outside/docs/models/dino/dino-sdss_smote-10-05-2026-sdss_smote                                 |
|      3 | dino         | decals_raw                |     0.696  |              0.6662 |     0.6745 |        0.6561 |                             0.6571 |          0.9462 |     0.8975 |                   809 |               1219 | /home/alexmrtr/Pucpr/a-look-outside/docs/models/dino/dino-decals_raw-10-05-2026-decals_raw                                 |
|      4 | dino         | sdss_raw                  |     0.8133 |              0.6395 |     0.6406 |        0.759  |                             0.7599 |          0.9711 |     0.556  |                   610 |               1672 | /home/alexmrtr/Pucpr/a-look-outside/docs/models/dino/dino-sdss_raw-10-05-2026-sdss_raw                                     |

## Best model per dataset (by balanced_accuracy)

| dataset_name              | best_model   |   balanced_accuracy | runner_up   | gap   |
|:--------------------------|:-------------|--------------------:|:------------|:------|
| decals_raw                | dino         |              0.6662 |             | n/a   |
| sdss_random_over_sampling | dino         |              0.9764 |             | n/a   |
| sdss_raw                  | dino         |              0.6395 |             | n/a   |
| sdss_smote                | dino         |              0.9376 |             | n/a   |

## Best dataset per model (by balanced_accuracy)

| model_name   | best_dataset              |   balanced_accuracy |
|:-------------|:--------------------------|--------------------:|
| dino         | sdss_random_over_sampling |              0.9764 |

## Model rankings by metric

### accuracy

|   rank | model_name   | dataset_name              |   accuracy |
|-------:|:-------------|:--------------------------|-----------:|
|      1 | dino         | sdss_random_over_sampling |     0.9764 |
|      2 | dino         | sdss_smote                |     0.9376 |
|      3 | dino         | sdss_raw                  |     0.8133 |
|      4 | dino         | decals_raw                |     0.696  |

### balanced_accuracy

|   rank | model_name   | dataset_name              |   balanced_accuracy |
|-------:|:-------------|:--------------------------|--------------------:|
|      1 | dino         | sdss_random_over_sampling |              0.9764 |
|      2 | dino         | sdss_smote                |              0.9376 |
|      3 | dino         | decals_raw                |              0.6662 |
|      4 | dino         | sdss_raw                  |              0.6395 |

### macro_f1

|   rank | model_name   | dataset_name              |   macro_f1 |
|-------:|:-------------|:--------------------------|-----------:|
|      1 | dino         | sdss_random_over_sampling |     0.9763 |
|      2 | dino         | sdss_smote                |     0.9378 |
|      3 | dino         | decals_raw                |     0.6745 |
|      4 | dino         | sdss_raw                  |     0.6406 |

### cohen_kappa

|   rank | model_name   | dataset_name              |   cohen_kappa |
|-------:|:-------------|:--------------------------|--------------:|
|      1 | dino         | sdss_random_over_sampling |        0.9737 |
|      2 | dino         | sdss_smote                |        0.9307 |
|      3 | dino         | sdss_raw                  |        0.759  |
|      4 | dino         | decals_raw                |        0.6561 |

### matthews_correlation_coefficient

|   rank | model_name   | dataset_name              |   matthews_correlation_coefficient |
|-------:|:-------------|:--------------------------|-----------------------------------:|
|      1 | dino         | sdss_random_over_sampling |                             0.9738 |
|      2 | dino         | sdss_smote                |                             0.9308 |
|      3 | dino         | sdss_raw                  |                             0.7599 |
|      4 | dino         | decals_raw                |                             0.6571 |

### roc_auc_macro

|   rank | model_name   | dataset_name              |   roc_auc_macro |
|-------:|:-------------|:--------------------------|----------------:|
|      1 | dino         | sdss_random_over_sampling |          0.9985 |
|      2 | dino         | sdss_smote                |          0.9962 |
|      3 | dino         | sdss_raw                  |          0.9711 |
|      4 | dino         | decals_raw                |          0.9462 |

### log_loss

|   rank | model_name   | dataset_name              |   log_loss |
|-------:|:-------------|:--------------------------|-----------:|
|      1 | dino         | sdss_random_over_sampling |     0.1162 |
|      2 | dino         | sdss_smote                |     0.2206 |
|      3 | dino         | sdss_raw                  |     0.556  |
|      4 | dino         | decals_raw                |     0.8975 |


## Recent runs (chronological, last 20)

| completed_at              | model_name   | dataset_name              |   accuracy |   balanced_accuracy |   duration_seconds |
|:--------------------------|:-------------|:--------------------------|-----------:|--------------------:|-------------------:|
| 2026-05-11T03:13:14+00:00 | dino         | sdss_random_over_sampling |     0.9764 |              0.9764 |               5720 |
| 2026-05-11T01:12:08+00:00 | dino         | sdss_smote                |     0.9376 |              0.9376 |               6382 |
| 2026-05-10T23:25:38+00:00 | dino         | sdss_raw                  |     0.8133 |              0.6395 |               1672 |
| 2026-05-10T22:57:42+00:00 | dino         | decals_raw                |     0.696  |              0.6662 |               1219 |