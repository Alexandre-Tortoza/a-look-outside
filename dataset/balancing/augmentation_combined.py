from __future__ import annotations

import albumentations as A
import numpy as np


def apply(
    images: np.ndarray,
    labels: np.ndarray,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    transform = A.Compose([
        # geometry
        A.Rotate(limit=180, border_mode=4, p=0.9),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=0,
            border_mode=4,
            p=0.4,
        ),
        # photometry
        A.RandomBrightnessContrast(
            brightness_limit=0.25,
            contrast_limit=0.25,
            p=0.6,
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.2),
        # noise / blur
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.04), p=1.0),
            A.ISONoise(p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.4),
        # deformation
        A.OneOf([
            A.ElasticTransform(alpha=30.0, sigma=5.0, border_mode=4, p=1.0),
            A.GridDistortion(num_steps=4, distort_limit=0.2, border_mode=4, p=1.0),
        ], p=0.3),
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
