# sdss_smote — comparison across models

- Primary metric: **balanced_accuracy**
- Rows ordered by balanced_accuracy (desc).

| model_name   |   accuracy |   balanced_accuracy |   macro_f1 |   cohen_kappa |   matthews_correlation_coefficient |   roc_auc_macro |   log_loss |   misclassified_count |
|:-------------|-----------:|--------------------:|-----------:|--------------:|-----------------------------------:|----------------:|-----------:|----------------------:|
| efficientnet |     0.9721 |              0.9721 |     0.972  |        0.969  |                             0.969  |          0.9987 |     0.1229 |                   293 |
| resnet50     |     0.971  |              0.971  |     0.9708 |        0.9678 |                             0.9679 |          0.9985 |     0.1201 |                   304 |
| dino         |     0.9643 |              0.9643 |     0.964  |        0.9603 |                             0.9603 |          0.9935 |     0.1814 |                   375 |
| dino         |     0.9607 |              0.9606 |     0.9604 |        0.9563 |                             0.9563 |          0.9971 |     0.1683 |                   413 |
| dino         |     0.9376 |              0.9376 |     0.9378 |        0.9307 |                             0.9308 |          0.9962 |     0.2206 |                   655 |

See `comparison.png` for the visual comparison.
