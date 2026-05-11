# sdss_random_over_sampling — comparison across models

- Primary metric: **balanced_accuracy**
- Rows ordered by balanced_accuracy (desc).

| model_name   |   accuracy |   balanced_accuracy |   macro_f1 |   cohen_kappa |   matthews_correlation_coefficient |   roc_auc_macro |   log_loss |   misclassified_count |
|:-------------|-----------:|--------------------:|-----------:|--------------:|-----------------------------------:|----------------:|-----------:|----------------------:|
| dino         |     0.9764 |              0.9764 |     0.9763 |        0.9737 |                             0.9738 |          0.9985 |     0.1162 |                   248 |

See `comparison.png` for the visual comparison.
