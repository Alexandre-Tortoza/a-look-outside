from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from models._base import ModelAdapter
from models.dino import DinoAdapter
from models.efficientnet import EfficientNetAdapter
from models.k_nearest_neighbors import KNearestNeighborsAdapter
from models.resnet50 import ResNet50Adapter
from models.vgg16 import VGG16Adapter


@dataclass(frozen=True)
class ModelInfo:
    name: str
    display_name: str
    is_deep_learning: bool


_FACTORIES: dict[str, Callable[..., ModelAdapter]] = {
    "dino": lambda **_: DinoAdapter(),
    "vgg16": lambda **_: VGG16Adapter(),
    "efficientnet": lambda **_: EfficientNetAdapter(),
    "resnet50": lambda **_: ResNet50Adapter(),
    "k_nearest_neighbors": lambda mode="pixels", n_neighbors=5, **_: (
        KNearestNeighborsAdapter(mode=mode, n_neighbors=n_neighbors)
    ),
}

_INFO: dict[str, ModelInfo] = {
    "dino": ModelInfo("dino", "DINO ViT-S/16 (timm)", True),
    "vgg16": ModelInfo("vgg16", "VGG16 (torchvision ImageNet)", True),
    "efficientnet": ModelInfo(
        "efficientnet", "EfficientNet-B0 (torchvision ImageNet)", True
    ),
    "resnet50": ModelInfo("resnet50", "ResNet50 (torchvision ImageNet)", True),
    "k_nearest_neighbors": ModelInfo(
        "k_nearest_neighbors", "K-Nearest Neighbors", False
    ),
}


def list_models() -> list[ModelInfo]:
    return list(_INFO.values())


def get_model_info(name: str) -> ModelInfo:
    if name not in _INFO:
        available = ", ".join(_INFO)
        raise KeyError(f"unknown model '{name}'. available: {available}")
    return _INFO[name]


def build_adapter(name: str, **kwargs: Any) -> ModelAdapter:
    if name not in _FACTORIES:
        available = ", ".join(_FACTORIES)
        raise KeyError(f"unknown model '{name}'. available: {available}")
    return _FACTORIES[name](**kwargs)
