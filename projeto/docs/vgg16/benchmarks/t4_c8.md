# Resultados — VGG16

**Experimento:** vgg16

## Métricas Globais

| Métrica | Valor |
|---------|-------|
| Acurácia Top-1 | 0.9071 |
| Acurácia Top-5 | 0.9961 |
| Precisão Macro | 0.9010 |
| Recall Macro | 0.8903 |
| F1 Macro | 0.8880 |

## Acurácia por Classe

| ID | Classe | Acurácia |
|----|--------|----------|
| 0 | Disk, Face-on, No Bulge | 0.4653 |
| 1 | Smooth, Completely Round | 0.9682 |
| 2 | Smooth, In-between Round | 0.9490 |
| 3 | Smooth, Cigar Shaped | 0.9674 |
| 4 | Disk, Edge-on, Boxy Bulge | 0.9461 |
| 5 | Disk, Edge-on, No Bulge | 0.9388 |
| 6 | Disk, Edge-on, Rounded Bulge | 0.8360 |
| 7 | Disk, Face-on, Tight Spiral | 0.9136 |
| 8 | Disk, Face-on, Medium Spiral | 0.9705 |
| 9 | Disk, Face-on, Loose Spiral | 0.9482 |

## Relatório Completo

```
                              precision    recall  f1-score   support

     Disk, Face-on, No Bulge       0.88      0.47      0.61      1081
    Smooth, Completely Round       0.90      0.97      0.93      1853
    Smooth, In-between Round       0.97      0.95      0.96      2645
        Smooth, Cigar Shaped       0.94      0.97      0.95      2027
   Disk, Edge-on, Boxy Bulge       0.80      0.95      0.87       334
     Disk, Edge-on, No Bulge       0.92      0.94      0.93      2043
Disk, Edge-on, Rounded Bulge       0.90      0.84      0.86      1829
 Disk, Face-on, Tight Spiral       0.80      0.91      0.85      2628
Disk, Face-on, Medium Spiral       0.94      0.97      0.95      1423
 Disk, Face-on, Loose Spiral       0.98      0.95      0.96      1873

                    accuracy                           0.91     17736
                   macro avg       0.90      0.89      0.89     17736
                weighted avg       0.91      0.91      0.90     17736

```
