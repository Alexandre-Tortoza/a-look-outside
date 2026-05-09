from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import psutil
import torch
import yaml

GIGABYTE = 1024 ** 3


def detect_computer() -> dict[str, Any]:
    physical_cores = psutil.cpu_count(logical=False) or 1
    logical_cores = psutil.cpu_count(logical=True) or physical_cores
    total_memory_bytes = psutil.virtual_memory().total
    processor_name = _readable_processor_name()

    gpu_available = torch.cuda.is_available()
    gpu_name: str | None = None
    gpu_memory_gigabytes: float | None = None
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gigabytes = round(
            torch.cuda.get_device_properties(0).total_memory / GIGABYTE, 2
        )

    return {
        "processor_name": processor_name,
        "physical_core_count": physical_cores,
        "logical_core_count": logical_cores,
        "total_memory_gigabytes": round(total_memory_bytes / GIGABYTE, 2),
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "gpu_memory_gigabytes": gpu_memory_gigabytes,
    }


def default_resource_limits() -> dict[str, Any]:
    return {
        "use_gpu": True,
        "gpu_device": "auto",
        "maximum_gpu_memory_gigabytes": None,
        "maximum_cpu_worker_count": None,
        "maximum_memory_gigabytes": None,
        "use_mixed_precision": True,
    }


def load_or_create_configuration(path: Path) -> tuple[dict[str, Any], bool]:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}, False

    configuration = {
        "computer": detect_computer(),
        "resource_limits": default_resource_limits(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(configuration, handle, sort_keys=False)
    return configuration, True


def resolve_device(configuration: dict[str, Any]) -> torch.device:
    limits = configuration.get("resource_limits", {})
    computer = configuration.get("computer", {})
    if not limits.get("use_gpu", True) or not computer.get("gpu_available", False):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        return torch.device("cpu")
    requested = limits.get("gpu_device", "auto")
    if requested in (None, "auto"):
        return torch.device("cuda:0")
    if isinstance(requested, int):
        return torch.device(f"cuda:{requested}")
    return torch.device(str(requested))


def resolve_worker_count(configuration: dict[str, Any]) -> int:
    limits = configuration.get("resource_limits", {})
    computer = configuration.get("computer", {})
    logical_cores = int(computer.get("logical_core_count", 1) or 1)
    requested = limits.get("maximum_cpu_worker_count")
    if requested is None:
        return max(logical_cores - 1, 0)
    return min(int(requested), logical_cores)


def use_mixed_precision(configuration: dict[str, Any], device: torch.device) -> bool:
    limits = configuration.get("resource_limits", {})
    return bool(limits.get("use_mixed_precision", True)) and device.type == "cuda"


def _readable_processor_name() -> str:
    name = platform.processor() or platform.machine()
    cpuinfo_path = Path("/proc/cpuinfo")
    if cpuinfo_path.exists():
        with cpuinfo_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return name
