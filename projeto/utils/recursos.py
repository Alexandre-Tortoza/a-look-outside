"""Gestao automatica de recursos de hardware (GPU/CPU)."""

from __future__ import annotations

import multiprocessing
import os
from typing import Any

import torch


def configurar_dispositivo(config_recursos: dict[str, Any]) -> torch.device:
    """Seleciona o device e aplica limites de memoria se configurado.

    - ``"auto"``: usa CUDA se disponivel, senao CPU.
    - Se ``max_gpu_memoria_gb`` definido, aplica ``set_per_process_memory_fraction``.

    Returns:
        ``torch.device`` configurado e pronto para uso.
    """
    dispositivo_cfg = config_recursos.get("dispositivo", "auto")

    if dispositivo_cfg == "auto":
        dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dispositivo = torch.device(dispositivo_cfg)

    # Cap de memoria GPU
    if dispositivo.type == "cuda":
        max_gb = config_recursos.get("max_gpu_memoria_gb")
        if max_gb is not None:
            idx = dispositivo.index or 0
            total_bytes = torch.cuda.get_device_properties(idx).total_mem
            total_gb = total_bytes / (1024 ** 3)
            fracao = min(max_gb / total_gb, 1.0)
            torch.cuda.set_per_process_memory_fraction(fracao, idx)

    return dispositivo


def obter_num_workers(config_recursos: dict[str, Any]) -> int:
    """Retorna o numero de workers para o DataLoader.

    Se ``null`` no config, usa ``min(cpu_count, 8)``.
    """
    valor = config_recursos.get("num_workers")
    if valor is not None:
        return int(valor)
    return min(multiprocessing.cpu_count(), 8)


def aplicar_batch_cap(batch_size: int, config_recursos: dict[str, Any]) -> int:
    """Aplica cap de batch_size se ``max_batch_em_memoria`` estiver definido."""
    cap = config_recursos.get("max_batch_em_memoria")
    if cap is not None:
        return min(batch_size, int(cap))
    return batch_size


def usar_mixed_precision(config_recursos: dict[str, Any], dispositivo: torch.device) -> bool:
    """Retorna True se AMP deve ser usado (configurado e em GPU)."""
    return config_recursos.get("mixed_precision", True) and dispositivo.type == "cuda"


def info_recursos() -> dict[str, Any]:
    """Retorna informacoes sobre os recursos disponiveis na maquina.

    Returns:
        Dict com GPU, CPU e RAM info.
    """
    info: dict[str, Any] = {
        "cpu_count": multiprocessing.cpu_count(),
        "cuda_disponivel": torch.cuda.is_available(),
    }

    # RAM
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024 ** 3), 1)
        info["ram_disponivel_gb"] = round(mem.available / (1024 ** 3), 1)
    except ImportError:
        # Fallback via /proc/meminfo no Linux
        try:
            with open("/proc/meminfo") as f:
                linhas = f.readlines()
            for linha in linhas:
                if linha.startswith("MemTotal:"):
                    kb = int(linha.split()[1])
                    info["ram_total_gb"] = round(kb / (1024 ** 2), 1)
                elif linha.startswith("MemAvailable:"):
                    kb = int(linha.split()[1])
                    info["ram_disponivel_gb"] = round(kb / (1024 ** 2), 1)
        except Exception:
            info["ram_total_gb"] = "N/A"
            info["ram_disponivel_gb"] = "N/A"

    # GPU
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        info["gpus"] = []
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            total_gb = round(props.total_memory / (1024 ** 3), 1)
            livre_gb = round(torch.cuda.mem_get_info(i)[0] / (1024 ** 3), 1)
            info["gpus"].append({
                "indice": i,
                "nome": props.name,
                "vram_total_gb": total_gb,
                "vram_livre_gb": livre_gb,
            })
    else:
        info["gpus"] = []

    return info
