# Dataset analysis — sdss_smote

- Source path: `/home/alexmrtr/Pucpr/a-look-outside/dataset/processed/sdss_smote.h5`
- Sample count: **69970**
- Number of classes: **10**
- Image shape (H, W, C): [69, 69, 3]
- Channels: 3
- Image dtype: `uint8`
- Labels dtype: `int64`
- File size: 576.5 MB

## Class balance

- Minority class count: 6997
- Majority class count: 6997
- Imbalance ratio (majority / minority): 1.0000

## Intensity (normalised to [0, 1])

- Min: 0.0000
- Max: 1.0000
- Mean: 0.0764
- Std: 0.1174

## Per-channel statistics (normalised)

- **red**: mean=0.0898, std=0.1357
- **green**: mean=0.0779, std=0.1139
- **blue**: mean=0.0615, std=0.0977

## Artifacts

- `metrics.json`: full machine-readable summary.
- `class_statistics.csv`: per-class counts, percentages, channel statistics.
- `value_check.md`: integrity check report.
- `class_distribution.png`, `class_balance.png`
- `pixel_intensity_histogram.png`
- `sample_mosaic.png`
