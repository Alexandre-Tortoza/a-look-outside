from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE


def apply(
    images: np.ndarray,
    labels: np.ndarray,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    original_shape = images.shape[1:]
    original_dtype = images.dtype

    flattened = images.reshape(images.shape[0], -1).astype(np.float32, copy=False)

    minority_count = int(np.bincount(labels.astype(np.int64)).min())
    if minority_count <= 1:
        raise ValueError(
            "SMOTE requires at least 2 samples in the smallest class "
            f"(found {minority_count})"
        )
    k_neighbors = min(5, minority_count - 1)

    sampler = SMOTE(random_state=random_seed, k_neighbors=k_neighbors)
    resampled_flat, resampled_labels = sampler.fit_resample(flattened, labels)

    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        resampled_flat = np.clip(resampled_flat, info.min, info.max)
    elif np.issubdtype(original_dtype, np.floating):
        if images.max() <= 1.0 + 1e-6 and images.min() >= -1e-6:
            resampled_flat = np.clip(resampled_flat, 0.0, 1.0)

    resampled_images = resampled_flat.reshape(-1, *original_shape).astype(original_dtype)
    return resampled_images, resampled_labels.astype(labels.dtype)
