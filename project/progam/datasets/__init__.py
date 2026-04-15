"""
Pipeline de balanceamento de datasets para classificação de galáxias.

Módulo responsável por carregar, processar e aplicar técnicas de balanceamento
em datasets de morfologia galáctica (SDSS e DECaLS).
"""

from .carregador import CarregadorDataset
from .pipelines import PipelineBalanceamento

__all__ = [
    "CarregadorDataset",
    "PipelineBalanceamento",
]
