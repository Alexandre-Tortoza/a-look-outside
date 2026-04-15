"""
Detecção automática de hardware disponível.

Inspeciona CPU, memória RAM e GPU para sugerir configurações otimizadas.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class InfoCPU:
    nome: str
    nucleos_fisicos: int
    nucleos_logicos: int
    frequencia_mhz: float


@dataclass
class InfoMemoria:
    total_gb: float
    disponivel_gb: float
    percentual_uso: float


@dataclass
class InfoGPU:
    indice: int
    nome: str
    memoria_total_gb: float
    memoria_disponivel_gb: float
    driver: str = ""


@dataclass
class PerfilHardware:
    cpu: InfoCPU
    memoria: InfoMemoria
    gpus: List[InfoGPU] = field(default_factory=list)

    @property
    def tem_gpu(self) -> bool:
        return len(self.gpus) > 0

    @property
    def dispositivo_recomendado(self) -> str:
        if self.tem_gpu:
            return f"cuda:{self.gpus[0].indice}"
        return "cpu"

    @property
    def workers_recomendados(self) -> int:
        """Número de workers para DataLoader baseado em núcleos disponíveis."""
        return min(self.cpu.nucleos_fisicos, 8)

    @property
    def batch_size_recomendado(self) -> int:
        """Batch size baseado na memória disponível."""
        if self.tem_gpu:
            mem_gb = self.gpus[0].memoria_disponivel_gb
        else:
            mem_gb = self.memoria.disponivel_gb

        if mem_gb >= 8:
            return 64
        elif mem_gb >= 4:
            return 32
        elif mem_gb >= 2:
            return 16
        return 8


def detectar_hardware() -> PerfilHardware:
    """
    Detectar especificações completas do hardware.

    Returns:
        PerfilHardware com CPU, memória e GPUs disponíveis.
    """
    cpu = _detectar_cpu()
    memoria = _detectar_memoria()
    gpus = _detectar_gpus()

    return PerfilHardware(cpu=cpu, memoria=memoria, gpus=gpus)


def _detectar_cpu() -> InfoCPU:
    """Detectar informações do CPU."""
    frequencia = psutil.cpu_freq()
    freq_mhz = frequencia.current if frequencia else 0.0

    return InfoCPU(
        nome=_nome_cpu(),
        nucleos_fisicos=psutil.cpu_count(logical=False) or 1,
        nucleos_logicos=psutil.cpu_count(logical=True) or 1,
        frequencia_mhz=freq_mhz,
    )


def _nome_cpu() -> str:
    """Obter nome do CPU via /proc/cpuinfo."""
    try:
        with open("/proc/cpuinfo") as f:
            for linha in f:
                if "model name" in linha:
                    return linha.split(":")[1].strip()
    except OSError:
        pass
    return os.uname().machine


def _detectar_memoria() -> InfoMemoria:
    """Detectar informações de memória RAM."""
    mem = psutil.virtual_memory()
    return InfoMemoria(
        total_gb=mem.total / (1024 ** 3),
        disponivel_gb=mem.available / (1024 ** 3),
        percentual_uso=mem.percent,
    )


def _detectar_gpus() -> List[InfoGPU]:
    """Detectar GPUs disponíveis via torch.cuda ou nvidia-smi."""
    gpus = []

    try:
        import torch

        if not torch.cuda.is_available():
            return gpus

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_total = props.total_memory / (1024 ** 3)

            try:
                torch.cuda.set_device(i)
                mem_livre = (
                    props.total_memory - torch.cuda.memory_allocated(i)
                ) / (1024 ** 3)
            except Exception:
                mem_livre = mem_total

            gpus.append(
                InfoGPU(
                    indice=i,
                    nome=props.name,
                    memoria_total_gb=mem_total,
                    memoria_disponivel_gb=mem_livre,
                )
            )

    except ImportError:
        logger.debug("torch não disponível para detecção de GPU")

    return gpus
