# DINO ViT-S/16 (timm) on sdss_raw

- Run directory: `/home/alexmrtr/Pucpr/a-look-outside/machine-learning/runs/dino-sdss_raw-10-05-2026`
- Deep learning: yes
- Train/val/test sizes: 15249/3268/3268
- Image size: 224
- Number of classes: 10

## Headline metrics

- Test accuracy: **0.8133**
- Balanced accuracy: **0.6395**
- Macro F1: **0.6406**
- Weighted F1: **0.8081**
- Cohen kappa: **0.7590**
- MCC: **0.7599**
- ROC-AUC (macro): **0.9711**
- Average precision (macro): **0.6825**
- Log loss: **0.5560**
- top-3 accuracy: **0.9816**
- top-5 accuracy: **0.9979**
- Misclassified samples: **610**

## Training configuration

- Epoch budget: 50
- Early stopping patience: 8
- Batch size: 32
- Learning rate: 0.0001
- Random seed: 42

## Artifacts

- `metrics.md`: aggregate metrics table.
- `classification_report.md`: per-class precision/recall/F1.
- `confusion_matrix.png`, `confusion_matrix_normalized.png`
- `per_class_accuracy.png`
- `class_distribution.png`
- `roc_curves.png`, `precision_recall_curves.png` (when probabilities available)
- `error_analysis.md` + `misclassified_samples/` in run directory
- `predictions.csv`, `misclassified.csv` in run directory
- `learning_curves.png`
- `calibration_plot.png`
