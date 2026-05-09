from __future__ import annotations

import numpy as np
from imblearn.over_sampling import RandomOverSampler


def apply(
    images: np.ndarray,
    labels: np.ndarray,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    original_shape = images.shape[1:]
    original_dtype = images.dtype

    flattened = images.reshape(images.shape[0], -1)

    sampler = RandomOverSampler(random_state=random_seed)
    resampled_flat, resampled_labels = sampler.fit_resample(flattened, labels)

    resampled_images = resampled_flat.reshape(-1, *original_shape).astype(original_dtype)
    return resampled_images, resampled_labels.astype(labels.dtype)
