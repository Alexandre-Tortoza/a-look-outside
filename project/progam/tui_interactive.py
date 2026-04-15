#!/usr/bin/env python
"""
Script de entrada para a TUI interativa.

Executa a interface de usuário em texto para configuração de experimentos.
"""

import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("tui.log"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


def main():
    """Função principal."""
    logger.info("Iniciando Interface Interativa...")

    try:
        from tui.app import executar

        logger.info("TUI carregada com sucesso")
        executar()

    except ImportError as e:
        logger.error(f"Erro ao importar TUI: {e}")
        logger.error("Certifique-se de que 'textual' está instalado")
        logger.error("Execute: pip install textual psutil")
        sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Aplicação encerrada pelo usuário")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Erro fatal: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
