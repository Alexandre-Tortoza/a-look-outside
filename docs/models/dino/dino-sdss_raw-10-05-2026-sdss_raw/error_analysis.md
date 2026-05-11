# Error analysis — DINO ViT-S/16 (timm) on sdss_raw

## Per-class accuracy

| class                      |   accuracy |
|:---------------------------|-----------:|
| disk_edge_on_boxy_bulge    |     0      |
| smooth_cigar_shaped        |     0.4808 |
| disk_face_on_loose_spiral  |     0.5    |
| disk_face_on_no_spiral     |     0.5472 |
| disk_face_on_medium_spiral |     0.6397 |
| disk_face_on_tight_spiral  |     0.6488 |
| disk_edge_on_no_bulge      |     0.8409 |
| disk_edge_on_rounded_bulge |     0.9    |
| smooth_in_between_round    |     0.9068 |
| smooth_completely_round    |     0.9305 |

## Most-confused pairs (true → predicted)

| true_class                | predicted_class            |   count |
|:--------------------------|:---------------------------|--------:|
| disk_face_on_no_spiral    | smooth_in_between_round    |      73 |
| disk_face_on_no_spiral    | smooth_completely_round    |      67 |
| smooth_completely_round   | smooth_in_between_round    |      46 |
| smooth_in_between_round   | smooth_completely_round    |      41 |
| disk_face_on_no_spiral    | disk_face_on_medium_spiral |      36 |
| disk_face_on_no_spiral    | disk_face_on_tight_spiral  |      33 |
| smooth_in_between_round   | disk_face_on_no_spiral     |      29 |
| disk_face_on_tight_spiral | disk_face_on_medium_spiral |      25 |
| disk_face_on_tight_spiral | disk_face_on_no_spiral     |      24 |
| smooth_completely_round   | disk_face_on_no_spiral     |      22 |

## Summary

- Total misclassified samples: **610**

- Misclassified samples saved at `dino-sdss_raw-10-05-2026/misclassified_samples/`

- Full predictions: `dino-sdss_raw-10-05-2026/predictions.csv`

- Errors only: `dino-sdss_raw-10-05-2026/misclassified.csv`
