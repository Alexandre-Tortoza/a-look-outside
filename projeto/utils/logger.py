"""Factory de loggers padronizados com saída para console e arquivo."""

import logging
import sys
from pathlib import Path
from typing import Optional

_FORMATO = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATA_FMT = "%Y-%m-%d %H:%M:%S"

try:
    import colorama  # type: ignore

    colorama.init(autoreset=True)
    _CORES = {
        "DEBUG": colorama.Fore.CYAN,
        "INFO": colorama.Fore.GREEN,
        "WARNING": colorama.Fore.YELLOW,
        "ERROR": colorama.Fore.RED,
        "CRITICAL": colorama.Fore.MAGENTA,
    }
    _RESET = colorama.Style.RESET_ALL
    _TEM_COLORAMA = True
except ImportError:
    _TEM_COLORAMA = False


class _FormatterColorido(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if _TEM_COLORAMA:
            cor = _CORES.get(record.levelname, "")
            return f"{cor}{msg}{_RESET}"
        return msg


def obter_logger(
    nome: str,
    arquivo_log: Optional[Path] = None,
    nivel: int = logging.DEBUG,
) -> logging.Logger:
    """Cria (ou recupera) um logger com handlers de console e opcionalmente de arquivo.

    Args:
        nome: Nome do logger (geralmente __name__ do módulo chamador).
        arquivo_log: Caminho do arquivo de log. Se None, apenas console.
        nivel: Nível mínimo de log. Console usa INFO; arquivo usa nivel.

    Returns:
        Logger configurado.
    """
    logger = logging.getLogger(nome)
    if logger.handlers:
        return logger

    logger.setLevel(nivel)
    logger.propagate = False

    # Handler de console — INFO+
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(_FormatterColorido(_FORMATO, datefmt=_DATA_FMT))
    logger.addHandler(ch)

    # Handler de arquivo — nível configurado
    if arquivo_log is not None:
        arquivo_log.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(arquivo_log, encoding="utf-8")
        fh.setLevel(nivel)
        fh.setFormatter(logging.Formatter(_FORMATO, datefmt=_DATA_FMT))
        logger.addHandler(fh)

    return logger


def configurar_logger_global(nivel: int = logging.INFO) -> None:
    """Configura o logger raiz. Chamar uma única vez no início do processo."""
    logging.basicConfig(
        level=nivel,
        format=_FORMATO,
        datefmt=_DATA_FMT,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
