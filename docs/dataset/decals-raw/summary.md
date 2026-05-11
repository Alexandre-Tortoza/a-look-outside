# Dataset analysis — decals-raw

- Source path: `/home/alexmrtr/Pucpr/a-look-outside/dataset/raw/decals.h5`
- Sample count: **17736**
- Number of classes: **10**
- Image shape (H, W, C): [256, 256, 3]
- Channels: 3
- Image dtype: `uint8`
- Labels dtype: `int64`
- File size: 2608.55 MB

## Class balance

- Minority class count: 334
- Majority class count: 2645
- Imbalance ratio (majority / minority): 7.9192

## Intensity (normalised to [0, 1])

- Min: 0.0000
- Max: 1.0000
- Mean: 0.1630
- Std: 0.1197

## Per-channel statistics (normalised)

- **red**: mean=0.1675, std=0.1287
- **green**: mean=0.1626, std=0.1180
- **blue**: mean=0.1589, std=0.1116

## Artifacts

- `metrics.json`: full machine-readable summary.
- `class_statistics.csv`: per-class counts, percentages, channel statistics.
- `value_check.md`: integrity check report.
- `class_distribution.png`, `class_balance.png`
- `pixel_intensity_histogram.png`
- `sample_mosaic.png`
