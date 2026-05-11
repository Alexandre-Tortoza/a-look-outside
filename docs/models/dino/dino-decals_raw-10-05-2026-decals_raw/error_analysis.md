# Error analysis — DINO ViT-S/16 (timm) on decals_raw

## Per-class accuracy

| class                   |   accuracy |
|:------------------------|-----------:|
| disturbed               |     0.2778 |
| unbarred_loose_spiral   |     0.5558 |
| cigar_shaped_smooth     |     0.56   |
| barred_spiral           |     0.6515 |
| merging                 |     0.6655 |
| unbarred_tight_spiral   |     0.7018 |
| edge_on_no_bulge        |     0.7277 |
| in_between_round_smooth |     0.8059 |
| edge_on_with_bulge      |     0.8541 |
| round_smooth            |     0.8615 |

## Most-confused pairs (true → predicted)

| true_class              | predicted_class       |   count |
|:------------------------|:----------------------|--------:|
| unbarred_loose_spiral   | unbarred_tight_spiral |      90 |
| disturbed               | unbarred_loose_spiral |      58 |
| barred_spiral           | unbarred_loose_spiral |      48 |
| unbarred_tight_spiral   | unbarred_loose_spiral |      43 |
| merging                 | unbarred_loose_spiral |      31 |
| barred_spiral           | unbarred_tight_spiral |      31 |
| round_smooth            | unbarred_tight_spiral |      25 |
| edge_on_no_bulge        | edge_on_with_bulge    |      25 |
| unbarred_loose_spiral   | disturbed             |      24 |
| in_between_round_smooth | unbarred_loose_spiral |      21 |

## Summary

- Total misclassified samples: **809**

- Misclassified samples saved at `dino-decals_raw-10-05-2026/misclassified_samples/`

- Full predictions: `dino-decals_raw-10-05-2026/predictions.csv`

- Errors only: `dino-decals_raw-10-05-2026/misclassified.csv`
