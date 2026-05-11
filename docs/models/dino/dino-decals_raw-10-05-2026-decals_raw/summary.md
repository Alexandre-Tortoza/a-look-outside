# DINO ViT-S/16 (timm) on decals_raw

- Run directory: `/home/alexmrtr/Pucpr/a-look-outside/machine-learning/runs/dino-decals_raw-10-05-2026`
- Deep learning: yes
- Train/val/test sizes: 12415/2660/2661
- Image size: 224
- Number of classes: 10

## Headline metrics

- Test accuracy: **0.6960**
- Balanced accuracy: **0.6662**
- Macro F1: **0.6745**
- Weighted F1: **0.6955**
- Cohen kappa: **0.6561**
- MCC: **0.6571**
- ROC-AUC (macro): **0.9462**
- Average precision (macro): **0.7351**
- Log loss: **0.8975**
- top-3 accuracy: **0.9252**
- top-5 accuracy: **0.9790**
- Misclassified samples: **809**

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
