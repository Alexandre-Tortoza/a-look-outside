from __future__ import annotations

import albumentations as A
import numpy as np


def apply(
    images: np.ndarray,
    labels: np.ndarray,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    transform = A.Compose([
        A.ElasticTransform(
            alpha=40.0,
            sigma=6.0,
            border_mode=4,  # REFLECT
            p=0.7,
        ),
        A.GridDistortion(
            num_steps=5,
            distort_limit=0.3,
            border_mode=4,
            p=0.5,
        ),
    ])

    rng = np.random.default_rng(random_seed)
    counts = np.bincount(labels.astype(np.int64))
    target = int(counts.max())

    new_images = list(images)
    new_labels = list(labels)

    for class_idx, count in enumerate(counts):
        if count == 0 or count >= target:
            continue
        class_images = images[labels == class_idx]
        indices = rng.integers(0, len(class_images), size=target - count)
        for idx in indices:
            aug = transform(image=class_images[idx])["image"]
            new_images.append(aug)
            new_labels.append(class_idx)

    arr_images = np.stack(new_images)
    arr_labels = np.array(new_labels, dtype=labels.dtype)
    perm = rng.permutation(len(arr_images))
    return arr_images[perm], arr_labels[perm]
