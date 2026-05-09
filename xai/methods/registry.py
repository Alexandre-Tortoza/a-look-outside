from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MACHINE_LEARNING_ROOT = PROJECT_ROOT / "machine-learning"
for path in (str(PROJECT_ROOT), str(MACHINE_LEARNING_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from models.registry import ModelInfo  # noqa: E402

from xai.methods import (  # noqa: E402
    gradient_class_activation_mapping,
    nearest_neighbors,
)


@dataclass(frozen=True)
class XaiMethod:
    name: str
    display_name: str
    applies_to: Callable[[ModelInfo], bool]
    apply: Callable[..., None]


_METHODS: dict[str, XaiMethod] = {
    "gradient_class_activation_mapping": XaiMethod(
        name="gradient_class_activation_mapping",
        display_name="Grad-CAM (deep learning models)",
        applies_to=lambda info: info.is_deep_learning,
        apply=gradient_class_activation_mapping.apply,
    ),
    "nearest_neighbors": XaiMethod(
        name="nearest_neighbors",
        display_name="Nearest neighbors (KNN models)",
        applies_to=lambda info: info.name == "k_nearest_neighbors",
        apply=nearest_neighbors.apply,
    ),
}


def list_methods() -> list[XaiMethod]:
    return list(_METHODS.values())


def get_method(name: str) -> XaiMethod:
    if name not in _METHODS:
        available = ", ".join(_METHODS)
        raise KeyError(f"unknown XAI method '{name}'. available: {available}")
    return _METHODS[name]
