# Dataset analysis — sdss_random_over_sampling

- Source path: `/home/alexmrtr/Pucpr/a-look-outside/dataset/processed/sdss_random_over_sampling.h5`
- Sample count: **69970**
- Number of classes: **10**
- Image shape (H, W, C): [69, 69, 3]
- Channels: 3
- Image dtype: `uint8`
- Labels dtype: `int64`
- File size: 394.3 MB

## Class balance

- Minority class count: 6997
- Majority class count: 6997
- Imbalance ratio (majority / minority): 1.0000

## Intensity (normalised to [0, 1])

- Min: 0.0000
- Max: 1.0000
- Mean: 0.0798
- Std: 0.1213

## Per-channel statistics (normalised)

- **red**: mean=0.0936, std=0.1399
- **green**: mean=0.0812, std=0.1173
- **blue**: mean=0.0647, std=0.1021

## Artifacts

- `metrics.json`: full machine-readable summary.
- `class_statistics.csv`: per-class counts, percentages, channel statistics.
- `value_check.md`: integrity check report.
- `class_distribution.png`, `class_balance.png`
- `pixel_intensity_histogram.png`
- `sample_mosaic.png`
