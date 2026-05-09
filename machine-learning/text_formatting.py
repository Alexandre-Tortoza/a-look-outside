from __future__ import annotations

from typing import Any


def format_metric(value: Any, decimals: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)
