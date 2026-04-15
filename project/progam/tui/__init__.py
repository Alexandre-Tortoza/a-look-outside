"""
Interface de Usuário em Texto (TUI) para Galaxy Morphology Classification.

Fornece interface interativa para:
- Seleção de datasets
- Seleção de modelos
- Configuração dinâmica de parâmetros
- Detecção automática de hardware
- Execução de experimentos
"""

from .app import ConfiguracaoExperimento, executar, criar_app

__all__ = [
    "ConfiguracaoExperimento",
    "executar",
    "criar_app",
]
