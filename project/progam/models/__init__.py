"""Model registry for galaxy morphology classification benchmark."""

from models.cnn import CNNLight, CNNRobust
from models.vit import ViTLight, ViTRobust
from models.mobilenet import MobileNetLight, MobileNetRobust
from models.easynet import EasyNet, EasyNetRobust

MODEL_REGISTRY = {
    "cnn_light": CNNLight,
    "cnn_robust": CNNRobust,
    "vit_light": ViTLight,
    "vit_robust": ViTRobust,
    "mobilenet_light": MobileNetLight,
    "mobilenet_robust": MobileNetRobust,
    "easynet_light": EasyNet,
    "easynet_robust": EasyNetRobust,
}


def get_model(name: str, **kwargs):
    """
    Instantiate a model by name.

    Args:
        name: Model identifier (e.g., "cnn_light", "vit_robust")
        **kwargs: Additional arguments passed to model constructor

    Returns:
        Instance of the requested model

    Raises:
        ValueError: If model name not found in registry
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model: {name}. Available: {available}")
    return MODEL_REGISTRY[name](**kwargs)


def list_models():
    """Return list of available model names."""
    return sorted(MODEL_REGISTRY.keys())
