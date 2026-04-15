"""
Serviços auxiliares para a TUI.

Fornece detecção de hardware, gerenciamento de datasets e modelos.
"""

from .detecao_hardware import PerfilHardware, detectar_hardware
from .gerenciador_datasets import GerenciadorDatasets, InfoDataset, TipoDataset
from .gerenciador_modelos import GerenciadorModelos, InfoModelo

__all__ = [
    "detectar_hardware",
    "PerfilHardware",
    "GerenciadorDatasets",
    "InfoDataset",
    "TipoDataset",
    "GerenciadorModelos",
    "InfoModelo",
]
