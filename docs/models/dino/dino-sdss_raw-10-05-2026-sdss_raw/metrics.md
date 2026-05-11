# Metrics — DINO ViT-S/16 (timm) on sdss_raw

## Headline

| metric                           |   value |
|:---------------------------------|--------:|
| accuracy                         |  0.8133 |
| balanced_accuracy                |  0.6395 |
| cohen_kappa                      |  0.759  |
| matthews_correlation_coefficient |  0.7599 |
| log_loss                         |  0.556  |
| roc_auc_macro                    |  0.9711 |
| average_precision_macro          |  0.6825 |

## Averaged precision/recall/F1

| average   |   precision |   recall |   f1_score |
|:----------|------------:|---------:|-----------:|
| macro     |      0.6504 |   0.6395 |     0.6406 |
| weighted  |      0.8084 |   0.8133 |     0.8081 |
| micro     |      0.8133 |   0.8133 |     0.8133 |

## Top-k accuracy

| metric   |   value |
|:---------|--------:|
| top_3    |  0.9816 |
| top_5    |  0.9979 |

## Per-class ROC-AUC / AP

|   class_index |   roc_auc |   average_precision |
|--------------:|----------:|--------------------:|
|             0 |    0.9058 |              0.6977 |
|             1 |    0.9838 |              0.9568 |
|             2 |    0.976  |              0.9406 |
|             3 |    0.9849 |              0.6906 |
|             4 |    0.9905 |              0.9153 |
|             5 |    0.9678 |              0.0478 |
|             6 |    0.9963 |              0.8762 |
|             7 |    0.9657 |              0.6158 |
|             8 |    0.9725 |              0.5836 |
|             9 |    0.9674 |              0.501  |
