# decals_raw — comparison across models

- Primary metric: **balanced_accuracy**
- Rows ordered by balanced_accuracy (desc).

| model_name   |   accuracy |   balanced_accuracy |   macro_f1 |   cohen_kappa |   matthews_correlation_coefficient |   roc_auc_macro |   log_loss |   misclassified_count |
|:-------------|-----------:|--------------------:|-----------:|--------------:|-----------------------------------:|----------------:|-----------:|----------------------:|
| dino         |     0.8129 |              0.8049 |     0.7958 |        0.7887 |                             0.789  |          0.9552 |     0.7301 |                   498 |
| dino         |     0.7971 |              0.7859 |     0.7759 |        0.7711 |                             0.7714 |          0.9538 |     0.7867 |                   540 |
| resnet50     |     0.8023 |              0.7827 |     0.78   |        0.7771 |                             0.7779 |          0.969  |     0.6444 |                   526 |
| efficientnet |     0.7971 |              0.781  |     0.7763 |        0.7706 |                             0.7712 |          0.9691 |     0.6284 |                   540 |
| resnet50     |     0.7775 |              0.7737 |     0.7617 |        0.7496 |                             0.7518 |          0.9642 |     0.7282 |                   592 |
| dino         |     0.7854 |              0.77   |     0.7662 |        0.758  |                             0.7584 |          0.9681 |     0.6509 |                   571 |
| dino         |     0.696  |              0.6662 |     0.6745 |        0.6561 |                             0.6571 |          0.9462 |     0.8975 |                   809 |

See `comparison.png` for the visual comparison.
