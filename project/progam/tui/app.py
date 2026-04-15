"""
Aplicação TUI principal usando Textual.

Orquestra o fluxo interativo de configuração e execução de experimentos.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult

from .telas import (
    TelaBoasVindas,
    TelaSelecaoDataset,
    TelaSelecaoModelo,
    TelaConfiguracaoParametros,
    TelaConfirmacaoExperimento,
    TelaListarDatasets,
    TelaListarModelos,
    TelaInfoHardware,
    TelaGerarDataset,
    TelaExecutandoExperimento,
)

logger = logging.getLogger(__name__)


@dataclass
class ConfiguracaoExperimento:
    """Configuração de um experimento."""

    dataset_selecionado: Optional[str] = None
    modelo_selecionado: Optional[str] = None
    tecnicas_balanceamento: list = field(default_factory=list)
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    dispositivo: str = "auto"
    num_workers: int = 4
    early_stop_ativo: bool = True
    early_stop_paciencia: int = 5
    modo_vit: str = "scratch"
    salvar_checkpoints: bool = True
    divisao_treino: float = 0.7

    def esta_completa(self) -> bool:
        """Verificar se configuração está completa."""
        return self.dataset_selecionado is not None and self.modelo_selecionado is not None


CSS = """
Screen {
    background: $surface;
    color: $text;
}

#titulo {
    width: 100%;
    height: auto;
    content-align: center middle;
    background: $accent;
    color: $text;
    text-style: bold;
    padding: 1 2;
    margin-bottom: 1;
}

#menu {
    width: 100%;
    align: center middle;
}

Button {
    margin: 1;
    width: 30;
}

Input {
    margin: 1;
}

#content {
    height: 1fr;
    layout: vertical;
}

#container {
    width: 100%;
    height: 1fr;
    border: solid $accent;
    padding: 1;
}

#buttons {
    dock: bottom;
    height: auto;
    align: center middle;
}
"""


class AplicacaoGalaxy(App):
    """Aplicação TUI principal para Galaxy Morphology Classification."""

    TITLE = "Galaxy Morphology Classification Benchmark"
    SUB_TITLE = "Interactive Training Configuration"
    CSS = CSS

    SCREENS = {
        "boas_vindas": TelaBoasVindas,
        "selecao_dataset": TelaSelecaoDataset,
        "selecao_modelo": TelaSelecaoModelo,
        "configuracao_parametros": TelaConfiguracaoParametros,
        "confirmacao_experimento": TelaConfirmacaoExperimento,
        "listar_datasets": TelaListarDatasets,
        "listar_modelos": TelaListarModelos,
        "info_hardware": TelaInfoHardware,
        "gerar_dataset": TelaGerarDataset,
        "executando_experimento": TelaExecutandoExperimento,
    }

    CSS_VARIABLES = {
        "surface": "#1e1e2e",
        "accent": "#7c3aed",
        "text": "#e0e0e0",
        "background": "#1e1e2e",
        "foreground": "#e0e0e0",
        "primary": "#7c3aed",
        "secondary": "#4c1d95",
        "warning": "#f59e0b",
        "error": "#ef4444",
        "success": "#10b981",
    }

    def __init__(self):
        super().__init__()
        self.config = ConfiguracaoExperimento()

    def on_mount(self) -> None:
        """Evento de montagem - exibir tela inicial."""
        self.push_screen("boas_vindas")


def criar_app() -> AplicacaoGalaxy:
    """Criar instância da aplicação."""
    return AplicacaoGalaxy()


def executar():
    """Executar aplicação TUI."""
    app = criar_app()
    app.run()
