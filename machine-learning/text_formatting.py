from __future__ import annotations

from typing import Any


def format_metric(value: Any, decimals: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric != numeric:
        return "n/a"
    return f"{numeric:.{decimals}f}"
