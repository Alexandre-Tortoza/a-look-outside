# Error analysis — DINO ViT-S/16 (timm) on sdss_smote

## Per-class accuracy

| class                      |   accuracy |
|:---------------------------|-----------:|
| disk_face_on_no_spiral     |     0.8275 |
| smooth_completely_round    |     0.8732 |
| smooth_in_between_round    |     0.8943 |
| disk_face_on_medium_spiral |     0.8951 |
| disk_face_on_loose_spiral  |     0.9581 |
| disk_face_on_tight_spiral  |     0.9676 |
| disk_edge_on_rounded_bulge |     0.9781 |
| smooth_cigar_shaped        |     0.9895 |
| disk_edge_on_no_bulge      |     0.9924 |
| disk_edge_on_boxy_bulge    |     1      |

## Most-confused pairs (true → predicted)

| true_class                 | predicted_class           |   count |
|:---------------------------|:--------------------------|--------:|
| smooth_completely_round    | smooth_in_between_round   |      77 |
| disk_face_on_medium_spiral | disk_face_on_tight_spiral |      77 |
| disk_face_on_no_spiral     | smooth_in_between_round   |      61 |
| smooth_in_between_round    | disk_face_on_no_spiral    |      50 |
| smooth_completely_round    | disk_face_on_no_spiral    |      48 |
| disk_face_on_no_spiral     | disk_face_on_tight_spiral |      46 |
| disk_face_on_no_spiral     | smooth_completely_round   |      45 |
| smooth_in_between_round    | smooth_completely_round   |      45 |
| disk_face_on_tight_spiral  | disk_face_on_no_spiral    |      27 |
| disk_face_on_medium_spiral | disk_face_on_no_spiral    |      25 |

## Summary

- Total misclassified samples: **655**

- Misclassified samples saved at `dino-sdss_smote-10-05-2026/misclassified_samples/`

- Full predictions: `dino-sdss_smote-10-05-2026/predictions.csv`

- Errors only: `dino-sdss_smote-10-05-2026/misclassified.csv`
