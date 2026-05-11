# Dataset analysis — sdss-raw

- Source path: `/home/alexmrtr/Pucpr/a-look-outside/dataset/raw/sdss.h5`
- Sample count: **21785**
- Number of classes: **10**
- Image shape (H, W, C): [69, 69, 3]
- Channels: 3
- Image dtype: `uint8`
- Labels dtype: `int64`
- File size: 200.5 MB

## Class balance

- Minority class count: 17
- Majority class count: 6997
- Imbalance ratio (majority / minority): 411.5882

## Intensity (normalised to [0, 1])

- Min: 0.0000
- Max: 1.0000
- Mean: 0.0911
- Std: 0.1267

## Per-channel statistics (normalised)

- **red**: mean=0.1086, std=0.1472
- **green**: mean=0.0934, std=0.1230
- **blue**: mean=0.0711, std=0.1032

## Artifacts

- `metrics.json`: full machine-readable summary.
- `class_statistics.csv`: per-class counts, percentages, channel statistics.
- `value_check.md`: integrity check report.
- `class_distribution.png`, `class_balance.png`
- `pixel_intensity_histogram.png`
- `sample_mosaic.png`
