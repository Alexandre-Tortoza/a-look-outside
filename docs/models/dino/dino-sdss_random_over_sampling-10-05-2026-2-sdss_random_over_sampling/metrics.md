# Metrics — DINOv2 ViT-S/14 (timm) on sdss_random_over_sampling

## Headline

| metric                           |   value |
|:---------------------------------|--------:|
| accuracy                         |  0.9764 |
| balanced_accuracy                |  0.9764 |
| cohen_kappa                      |  0.9737 |
| matthews_correlation_coefficient |  0.9738 |
| log_loss                         |  0.1162 |
| roc_auc_macro                    |  0.9985 |
| average_precision_macro          |  0.9904 |

## Averaged precision/recall/F1

| average   |   precision |   recall |   f1_score |
|:----------|------------:|---------:|-----------:|
| macro     |      0.9765 |   0.9764 |     0.9763 |
| weighted  |      0.9765 |   0.9764 |     0.9763 |
| micro     |      0.9764 |   0.9764 |     0.9764 |

## Top-k accuracy

| metric   |   value |
|:---------|--------:|
| top_3    |  0.9992 |
| top_5    |  0.9993 |

## Per-class ROC-AUC / AP

|   class_index |   roc_auc |   average_precision |
|--------------:|----------:|--------------------:|
|             0 |    0.9934 |              0.9684 |
|             1 |    0.9978 |              0.979  |
|             2 |    0.994  |              0.9584 |
|             3 |    1      |              1      |
|             4 |    1      |              0.9999 |
|             5 |    1      |              1      |
|             6 |    1      |              1      |
|             7 |    1      |              0.9997 |
|             8 |    0.9999 |              0.9993 |
|             9 |    0.9999 |              0.999  |
