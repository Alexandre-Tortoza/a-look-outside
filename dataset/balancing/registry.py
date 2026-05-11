from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from dataset.balancing import (
    augmentation_brightness,
    augmentation_combined,
    augmentation_elastic,
    augmentation_flip,
    augmentation_noise,
    augmentation_perspective,
    augmentation_rotation,
    random_over_sampling,
    random_under_sampling,
    smote,
)

BalancingFunction = Callable[[np.ndarray, np.ndarray, int], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class BalancingMethod:
    name: str
    display_name: str
    apply: BalancingFunction


_BALANCING_METHODS: dict[str, BalancingMethod] = {
    "smote": BalancingMethod(
        name="smote",
        display_name="SMOTE (synthetic minority oversampling)",
        apply=smote.apply,
    ),
    "random_over_sampling": BalancingMethod(
        name="random_over_sampling",
        display_name="Random over-sampling (duplicate minority classes)",
        apply=random_over_sampling.apply,
    ),
    "random_under_sampling": BalancingMethod(
        name="random_under_sampling",
        display_name="Random under-sampling (drop majority samples)",
        apply=random_under_sampling.apply,
    ),
    "augmentation_rotation": BalancingMethod(
        name="augmentation_rotation",
        display_name="Augmentation: rotation (0-360 degrees)",
        apply=augmentation_rotation.apply,
    ),
    "augmentation_flip": BalancingMethod(
        name="augmentation_flip",
        display_name="Augmentation: horizontal / vertical flip",
        apply=augmentation_flip.apply,
    ),
    "augmentation_brightness": BalancingMethod(
        name="augmentation_brightness",
        display_name="Augmentation: brightness and contrast jitter",
        apply=augmentation_brightness.apply,
    ),
    "augmentation_noise": BalancingMethod(
        name="augmentation_noise",
        display_name="Augmentation: Gaussian noise injection",
        apply=augmentation_noise.apply,
    ),
    "augmentation_elastic": BalancingMethod(
        name="augmentation_elastic",
        display_name="Augmentation: elastic deformation + grid distortion",
        apply=augmentation_elastic.apply,
    ),
    "augmentation_perspective": BalancingMethod(
        name="augmentation_perspective",
        display_name="Augmentation: perspective warp + shift/scale",
        apply=augmentation_perspective.apply,
    ),
    "augmentation_combined": BalancingMethod(
        name="augmentation_combined",
        display_name="Augmentation: combined (geometry + photometry + noise + deformation)",
        apply=augmentation_combined.apply,
    ),
}


def list_methods() -> list[BalancingMethod]:
    return list(_BALANCING_METHODS.values())


def get_method(name: str) -> BalancingMethod:
    if name not in _BALANCING_METHODS:
        available = ", ".join(_BALANCING_METHODS)
        raise KeyError(f"unknown balancing method '{name}'. available: {available}")
    return _BALANCING_METHODS[name]
