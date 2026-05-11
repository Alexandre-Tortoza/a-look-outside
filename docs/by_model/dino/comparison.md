# dino — comparison across datasets

- Primary metric: **balanced_accuracy**
- Rows ordered by balanced_accuracy (desc).

| dataset_name              |   accuracy |   balanced_accuracy |   macro_f1 |   cohen_kappa |   matthews_correlation_coefficient |   roc_auc_macro |   log_loss |   misclassified_count |
|:--------------------------|-----------:|--------------------:|-----------:|--------------:|-----------------------------------:|----------------:|-----------:|----------------------:|
| sdss_random_over_sampling |     0.9764 |              0.9764 |     0.9763 |        0.9737 |                             0.9738 |          0.9985 |     0.1162 |                   248 |
| sdss_smote                |     0.9376 |              0.9376 |     0.9378 |        0.9307 |                             0.9308 |          0.9962 |     0.2206 |                   655 |
| decals_raw                |     0.696  |              0.6662 |     0.6745 |        0.6561 |                             0.6571 |          0.9462 |     0.8975 |                   809 |
| sdss_raw                  |     0.8133 |              0.6395 |     0.6406 |        0.759  |                             0.7599 |          0.9711 |     0.556  |                   610 |

See `comparison.png` for the visual comparison.
