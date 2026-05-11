# DINOv2 ViT-S/14 (timm) on sdss_random_over_sampling

- Run directory: `/home/alexmrtr/Pucpr/a-look-outside/machine-learning/runs/dino-sdss_random_over_sampling-10-05-2026-2`
- Deep learning: yes
- Train/val/test sizes: 48979/10495/10496
- Image size: 224
- Number of classes: 10

## Headline metrics

- Test accuracy: **0.9764**
- Balanced accuracy: **0.9764**
- Macro F1: **0.9763**
- Weighted F1: **0.9763**
- Cohen kappa: **0.9737**
- MCC: **0.9738**
- ROC-AUC (macro): **0.9985**
- Average precision (macro): **0.9904**
- Log loss: **0.1162**
- top-3 accuracy: **0.9992**
- top-5 accuracy: **0.9993**
- Misclassified samples: **248**

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
