# sdss_raw — comparison across models

- Primary metric: **balanced_accuracy**
- Rows ordered by balanced_accuracy (desc).

| model_name   |   accuracy |   balanced_accuracy |   macro_f1 |   cohen_kappa |   matthews_correlation_coefficient |   roc_auc_macro |   log_loss |   misclassified_count |
|:-------------|-----------:|--------------------:|-----------:|--------------:|-----------------------------------:|----------------:|-----------:|----------------------:|
| dino         |     0.8133 |              0.6395 |     0.6406 |         0.759 |                             0.7599 |          0.9711 |      0.556 |                   610 |

See `comparison.png` for the visual comparison.
