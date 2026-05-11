# Metrics — DINO ViT-S/16 (timm) on sdss_smote

## Headline

| metric                           |   value |
|:---------------------------------|--------:|
| accuracy                         |  0.9376 |
| balanced_accuracy                |  0.9376 |
| cohen_kappa                      |  0.9307 |
| matthews_correlation_coefficient |  0.9308 |
| log_loss                         |  0.2206 |
| roc_auc_macro                    |  0.9962 |
| average_precision_macro          |  0.9741 |

## Averaged precision/recall/F1

| average   |   precision |   recall |   f1_score |
|:----------|------------:|---------:|-----------:|
| macro     |      0.9389 |   0.9376 |     0.9378 |
| weighted  |      0.9389 |   0.9376 |     0.9378 |
| micro     |      0.9376 |   0.9376 |     0.9376 |

## Top-k accuracy

| metric   |   value |
|:---------|--------:|
| top_3    |  0.9969 |
| top_5    |  0.9995 |

## Per-class ROC-AUC / AP

|   class_index |   roc_auc |   average_precision |
|--------------:|----------:|--------------------:|
|             0 |    0.9815 |              0.8986 |
|             1 |    0.995  |              0.9504 |
|             2 |    0.9929 |              0.9471 |
|             3 |    0.9999 |              0.9992 |
|             4 |    0.9996 |              0.9967 |
|             5 |    1      |              1      |
|             6 |    0.9999 |              0.9993 |
|             7 |    0.9975 |              0.9772 |
|             8 |    0.9969 |              0.9786 |
|             9 |    0.9992 |              0.9942 |
