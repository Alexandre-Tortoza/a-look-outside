"""
Balanceadores de datasets para tratamento de classes desbalanceadas.

Fornece diferentes estratégias de balanceamento:
- ADASYN: Adaptive Synthetic Sampling
- SMOTE: Synthetic Minority Over-sampling Technique
- Undersampling: Redução da classe majoritária
- Oversampling: Duplicação da classe minoritária
- Estratificação: Mantém proporções estratificadas
- Híbrido: Combinação de múltiplas técnicas
"""

from .adasyn import BalanceadorADASYN
from .base import BalanceadorBase
from .estratificacao import BalanceadorEstratificacao
from .hibrido import BalanceadorHibrido
from .oversampling import BalanceadorOversampling
from .smote import BalanceadorSMOTE
from .undersampling import BalanceadorUndersampling

__all__ = [
    "BalanceadorBase",
    "BalanceadorADASYN",
    "BalanceadorSMOTE",
    "BalanceadorUndersampling",
    "BalanceadorOversampling",
    "BalanceadorEstratificacao",
    "BalanceadorHibrido",
]
