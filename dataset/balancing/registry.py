from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from dataset.balancing import random_over_sampling, random_under_sampling, smote

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
}


def list_methods() -> list[BalancingMethod]:
    return list(_BALANCING_METHODS.values())


def get_method(name: str) -> BalancingMethod:
    if name not in _BALANCING_METHODS:
        available = ", ".join(_BALANCING_METHODS)
        raise KeyError(f"unknown balancing method '{name}'. available: {available}")
    return _BALANCING_METHODS[name]
