# Error analysis — DINOv2 ViT-S/14 (timm) on sdss_random_over_sampling

## Per-class accuracy

| class                      |   accuracy |
|:---------------------------|-----------:|
| disk_face_on_no_spiral     |     0.8923 |
| smooth_in_between_round    |     0.9248 |
| smooth_completely_round    |     0.9504 |
| disk_face_on_medium_spiral |     0.9981 |
| disk_face_on_tight_spiral  |     0.9981 |
| smooth_cigar_shaped        |     1      |
| disk_edge_on_rounded_bulge |     1      |
| disk_edge_on_boxy_bulge    |     1      |
| disk_edge_on_no_bulge      |     1      |
| disk_face_on_loose_spiral  |     1      |

## Most-confused pairs (true → predicted)

| true_class              | predicted_class            |   count |
|:------------------------|:---------------------------|--------:|
| disk_face_on_no_spiral  | smooth_in_between_round    |      52 |
| smooth_in_between_round | smooth_completely_round    |      48 |
| disk_face_on_no_spiral  | smooth_completely_round    |      42 |
| smooth_completely_round | smooth_in_between_round    |      37 |
| smooth_in_between_round | disk_face_on_no_spiral     |      25 |
| smooth_completely_round | disk_face_on_no_spiral     |      15 |
| disk_face_on_no_spiral  | disk_edge_on_rounded_bulge |       7 |
| disk_face_on_no_spiral  | disk_face_on_medium_spiral |       6 |
| disk_face_on_no_spiral  | disk_face_on_tight_spiral  |       3 |
| smooth_in_between_round | disk_face_on_tight_spiral  |       3 |

## Summary

- Total misclassified samples: **248**

- Misclassified samples saved at `dino-sdss_random_over_sampling-10-05-2026-2/misclassified_samples/`

- Full predictions: `dino-sdss_random_over_sampling-10-05-2026-2/predictions.csv`

- Errors only: `dino-sdss_random_over_sampling-10-05-2026-2/misclassified.csv`
