"""Registro central de todos os modelos do projeto."""

from __future__ import annotations

from modelos.base import ClassificadorGalaxias
from modelos.cnn.modelo import CNNBaseline
from modelos.dino.modelo import DinoGalaxy
from modelos.efficientnet.modelo import EfficientNetGalaxy
from modelos.multimodal.modelo import MultimodalGalaxy
from modelos.resnet50.modelo import ResNet50Galaxy
from modelos.vgg16.modelo import VGG16Galaxy
from modelos.vit.modelo import ViTGalaxy

_REGISTRO: dict[str, type[ClassificadorGalaxias]] = {
    "cnn": CNNBaseline,
    "resnet50": ResNet50Galaxy,
    "efficientnet": EfficientNetGalaxy,
    "vgg16": VGG16Galaxy,
    "vit": ViTGalaxy,
    "dino": DinoGalaxy,
    "multimodal": MultimodalGalaxy,
}


def obter_modelo(nome: str, **kwargs) -> ClassificadorGalaxias:
    """Instancia um modelo pelo nome.

    Args:
        nome: Nome registrado do modelo (case-insensitive).
        **kwargs: Argumentos passados ao construtor.

    Returns:
        Instância de ClassificadorGalaxias.

    Raises:
        ValueError: Se o nome não estiver no registro.
    """
    chave = nome.lower()
    if chave not in _REGISTRO:
        disponiveis = ", ".join(sorted(_REGISTRO.keys()))
        raise ValueError(f"Modelo '{nome}' não encontrado. Disponíveis: {disponiveis}")
    return _REGISTRO[chave](**kwargs)


def listar_modelos() -> list[str]:
    """Retorna lista ordenada de nomes de modelos registrados."""
    return sorted(_REGISTRO.keys())


__all__ = [
    "ClassificadorGalaxias",
    "CNNBaseline", "ResNet50Galaxy", "EfficientNetGalaxy", "VGG16Galaxy",
    "ViTGalaxy", "DinoGalaxy", "MultimodalGalaxy",
    "obter_modelo", "listar_modelos",
]
