# sdss_smote — comparison across models

- Primary metric: **balanced_accuracy**
- Rows ordered by balanced_accuracy (desc).

| model_name   |   accuracy |   balanced_accuracy |   macro_f1 |   cohen_kappa |   matthews_correlation_coefficient |   roc_auc_macro |   log_loss |   misclassified_count |
|:-------------|-----------:|--------------------:|-----------:|--------------:|-----------------------------------:|----------------:|-----------:|----------------------:|
| dino         |     0.9376 |              0.9376 |     0.9378 |        0.9307 |                             0.9308 |          0.9962 |     0.2206 |                   655 |

See `comparison.png` for the visual comparison.
