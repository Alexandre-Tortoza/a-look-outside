"""Engine genérico de treino, validação e checkpoint para todos os modelos."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from utils.logger import obter_logger
from utils.visualizacao import plotar_curva_treino


@dataclass
class HistoricoTreino:
    """Histórico completo de um experimento de treino."""

    perdas_treino: list[float] = field(default_factory=list)
    perdas_val: list[float] = field(default_factory=list)
    accs_treino: list[float] = field(default_factory=list)
    accs_val: list[float] = field(default_factory=list)
    melhor_val_acc: float = 0.0
    epocas_totais: int = 0
    parou_cedo: bool = False

    def resumo(self) -> str:
        return (
            f"Épocas: {self.epocas_totais} | "
            f"Melhor val_acc: {self.melhor_val_acc:.4f} | "
            f"Early stop: {self.parou_cedo}"
        )


class TreinadorModelo:
    """Engine de treino model-agnostic.

    Suporta:
    - AdamW + CosineAnnealingLR
    - Early stopping por val_acc
    - Checkpoint best-only (.pth)
    - AMP (mixed precision) quando CUDA disponível
    - Logging via logger injetado

    Args:
        epocas: Número máximo de épocas.
        lr: Learning rate inicial.
        dispositivo: "auto" | "cuda" | "cpu".
        dir_pesos: Diretório base para salvar checkpoints.
        dir_docs: Diretório base para salvar curvas de treino.
        paciencia_early_stop: Épocas sem melhora antes de parar.
        salvar_checkpoints: Se False, não salva nenhum .pth.
        scheduler_ativo: Se True, usa CosineAnnealingLR.
        peso_decay: Weight decay do AdamW.
        logger: Logger externo; se None, cria um interno.
    """

    def __init__(
        self,
        epocas: int = 50,
        lr: float = 1e-4,
        dispositivo: str = "auto",
        dir_pesos: Path = Path("pesos"),
        dir_docs: Path = Path("docs"),
        paciencia_early_stop: int = 10,
        salvar_checkpoints: bool = True,
        scheduler_ativo: bool = True,
        peso_decay: float = 1e-4,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.epocas = epocas
        self.lr = lr
        self.dir_pesos = Path(dir_pesos)
        self.dir_docs = Path(dir_docs)
        self.paciencia = paciencia_early_stop
        self.salvar_checkpoints = salvar_checkpoints
        self.scheduler_ativo = scheduler_ativo
        self.peso_decay = peso_decay
        self._log = logger or obter_logger(__name__)

        if dispositivo == "auto":
            self.dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.dispositivo = torch.device(dispositivo)

        self._usar_amp = self.dispositivo.type == "cuda"
        self._log.info("Dispositivo: %s | AMP: %s", self.dispositivo, self._usar_amp)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def treinar(
        self,
        rede: nn.Module,
        nome_experimento: str,
        loader_treino: DataLoader,
        loader_val: DataLoader,
    ) -> HistoricoTreino:
        """Executa o loop completo de treino.

        Args:
            rede: Modelo PyTorch a treinar.
            nome_experimento: Nome descritivo usado para nomear arquivos salvos.
            loader_treino: DataLoader do conjunto de treino.
            loader_val: DataLoader do conjunto de validação.

        Returns:
            HistoricoTreino com todas as métricas por época.
        """
        rede = rede.to(self.dispositivo)
        criterio = nn.CrossEntropyLoss()
        otimizador = AdamW(rede.parameters(), lr=self.lr, weight_decay=self.peso_decay)
        scheduler = (
            CosineAnnealingLR(otimizador, T_max=self.epocas, eta_min=1e-6)
            if self.scheduler_ativo
            else None
        )
        scaler = torch.cuda.amp.GradScaler() if self._usar_amp else None

        historico = HistoricoTreino()
        sem_melhora = 0
        caminho_pesos = self._caminho_pesos(nome_experimento)

        for epoca in range(1, self.epocas + 1):
            loss_t, acc_t = self._epoch_treino(rede, loader_treino, otimizador, criterio, scaler)
            loss_v, acc_v = self._epoch_validacao(rede, loader_val, criterio)

            if scheduler:
                scheduler.step()

            historico.perdas_treino.append(loss_t)
            historico.perdas_val.append(loss_v)
            historico.accs_treino.append(acc_t)
            historico.accs_val.append(acc_v)
            historico.epocas_totais = epoca

            self._log.info(
                "Época %3d/%d | loss %.4f | acc %.4f | val_loss %.4f | val_acc %.4f",
                epoca, self.epocas, loss_t, acc_t, loss_v, acc_v,
            )

            if acc_v > historico.melhor_val_acc:
                historico.melhor_val_acc = acc_v
                sem_melhora = 0
                if self.salvar_checkpoints:
                    self._salvar(rede, caminho_pesos)
                    self._log.info("  ✓ Melhor modelo salvo (val_acc=%.4f)", acc_v)
            else:
                sem_melhora += 1

            if sem_melhora >= self.paciencia:
                self._log.info("Early stopping após %d épocas sem melhora.", self.paciencia)
                historico.parou_cedo = True
                break

        # Salva curvas de treino
        dir_modelo = self.dir_docs / nome_experimento.split("_")[0]
        plotar_curva_treino(
            historico.perdas_treino,
            historico.perdas_val,
            historico.accs_treino,
            historico.accs_val,
            caminho_saida=dir_modelo / "curva_treino.png",
        )
        self._log.info("Treino concluído. %s", historico.resumo())
        return historico

    # ------------------------------------------------------------------
    # Utilitários internos
    # ------------------------------------------------------------------

    def _epoch_treino(
        self,
        rede: nn.Module,
        loader: DataLoader,
        otimizador: torch.optim.Optimizer,
        criterio: nn.Module,
        scaler,
    ) -> tuple[float, float]:
        rede.train()
        total_loss, total_acertos, total = 0.0, 0, 0

        for imgs, rots in loader:
            imgs = imgs.to(self.dispositivo, non_blocking=True)
            rots = rots.to(self.dispositivo, non_blocking=True)

            otimizador.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    logits = rede(imgs)
                    loss = criterio(logits, rots)
                scaler.scale(loss).backward()
                scaler.unscale_(otimizador)
                torch.nn.utils.clip_grad_norm_(rede.parameters(), 1.0)
                scaler.step(otimizador)
                scaler.update()
            else:
                logits = rede(imgs)
                loss = criterio(logits, rots)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(rede.parameters(), 1.0)
                otimizador.step()

            bs = rots.size(0)
            total_loss += loss.item() * bs
            total_acertos += (logits.argmax(dim=1) == rots).sum().item()
            total += bs

        return total_loss / total, total_acertos / total

    def _epoch_validacao(
        self,
        rede: nn.Module,
        loader: DataLoader,
        criterio: nn.Module,
    ) -> tuple[float, float]:
        rede.eval()
        total_loss, total_acertos, total = 0.0, 0, 0

        with torch.no_grad():
            for imgs, rots in loader:
                imgs = imgs.to(self.dispositivo, non_blocking=True)
                rots = rots.to(self.dispositivo, non_blocking=True)
                logits = rede(imgs)
                loss = criterio(logits, rots)
                bs = rots.size(0)
                total_loss += loss.item() * bs
                total_acertos += (logits.argmax(dim=1) == rots).sum().item()
                total += bs

        return total_loss / total, total_acertos / total

    def _caminho_pesos(self, nome_experimento: str) -> Path:
        nome_modelo = nome_experimento.split("_")[0]
        pasta = self.dir_pesos / nome_modelo
        pasta.mkdir(parents=True, exist_ok=True)
        return pasta / f"{nome_experimento}.pth"

    def _salvar(self, rede: nn.Module, caminho: Path) -> None:
        torch.save(rede.state_dict(), caminho)
