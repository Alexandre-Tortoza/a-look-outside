"""Motor de treinamento com checkpoints, early stopping e logging."""

import logging
from pathlib import Path
from typing import Callable, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class TreinadorModelo:
    """Orquestra o treinamento de um modelo com suporte a checkpoints e early stopping."""

    def __init__(
        self,
        num_epochs: int,
        learning_rate: float,
        dispositivo: str,
        dir_saida: Path,
        fn_log: Callable[[str], None] = None,
        fn_progresso: Callable[[int, int], None] = None,
        early_stop_ativo: bool = True,
        early_stop_paciencia: int = 5,
        salvar_checkpoints: bool = True,
    ):
        """
        Args:
            num_epochs: Número máximo de épocas
            learning_rate: Taxa de aprendizado
            dispositivo: 'cpu', 'cuda', ou 'auto'
            dir_saida: Diretório para salvar modelo e checkpoints
            fn_log: Callback para logging (recebe string)
            fn_progresso: Callback para progresso (recebe epoca_atual, total_epocas)
            early_stop_ativo: Se True, ativa early stopping
            early_stop_paciencia: Número de épocas sem melhora antes de parar
            salvar_checkpoints: Se True, salva checkpoint a cada época com melhora
        """
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.early_stop_ativo = early_stop_ativo
        self.early_stop_paciencia = early_stop_paciencia
        self.salvar_checkpoints = salvar_checkpoints
        self.dir_saida = Path(dir_saida)
        self.fn_log = fn_log or (lambda x: logger.info(x))
        self.fn_progresso = fn_progresso or (lambda e, t: None)

        if dispositivo == "auto":
            self.dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.dispositivo = dispositivo

        self.dir_saida.mkdir(parents=True, exist_ok=True)
        self.dir_checkpoints = self.dir_saida / "checkpoints"
        if self.salvar_checkpoints:
            self.dir_checkpoints.mkdir(exist_ok=True)

        self.fn_log(f"Dispositivo: {self.dispositivo}")

    def treinar(
        self,
        rede: nn.Module,
        nome_modelo: str,
        loader_treino: DataLoader,
        loader_val: DataLoader,
    ) -> dict:
        """
        Executa o treinamento.

        Args:
            rede: Modelo PyTorch
            nome_modelo: Nome do modelo (para salvar checkpoints)
            loader_treino: DataLoader de treinamento
            loader_val: DataLoader de validação

        Returns:
            Dict com métricas finais: accuracy, val_accuracy, loss, epocas_executadas
        """
        rede = rede.to(self.dispositivo)
        otimizador = AdamW(rede.parameters(), lr=self.learning_rate)
        criterio = nn.CrossEntropyLoss()

        melhor_val_acc = 0.0
        paciencia_contador = 0
        epocas_executadas = 0

        for epoca in range(1, self.num_epochs + 1):
            # ========== TREINAMENTO ==========
            rede.train()
            loss_acumulado = 0.0
            acertos_treino = 0
            total_treino = 0

            for imagens, rotulos in loader_treino:
                imagens = imagens.to(self.dispositivo)
                rotulos = rotulos.to(self.dispositivo)

                otimizador.zero_grad()
                logits = rede(imagens)
                loss = criterio(logits, rotulos)
                loss.backward()
                otimizador.step()

                loss_acumulado += loss.item() * imagens.size(0)
                _, previsoes = torch.max(logits, 1)
                acertos_treino += (previsoes == rotulos).sum().item()
                total_treino += rotulos.size(0)

            loss_medio = loss_acumulado / total_treino if total_treino > 0 else 0.0
            acc_treino = acertos_treino / total_treino if total_treino > 0 else 0.0

            # ========== VALIDAÇÃO ==========
            rede.eval()
            acertos_val = 0
            total_val = 0

            with torch.no_grad():
                for imagens, rotulos in loader_val:
                    imagens = imagens.to(self.dispositivo)
                    rotulos = rotulos.to(self.dispositivo)

                    logits = rede(imagens)
                    _, previsoes = torch.max(logits, 1)
                    acertos_val += (previsoes == rotulos).sum().item()
                    total_val += rotulos.size(0)

            acc_val = acertos_val / total_val if total_val > 0 else 0.0

            # Log
            log_msg = (
                f"Época {epoca}/{self.num_epochs} | "
                f"Loss: {loss_medio:.4f} | "
                f"Acc Train: {acc_treino:.2%} | "
                f"Acc Val: {acc_val:.2%}"
            )
            self.fn_log(log_msg)
            self.fn_progresso(epoca, self.num_epochs)
            epocas_executadas = epoca

            # ========== CHECKPOINT ==========
            if self.salvar_checkpoints and acc_val > melhor_val_acc:
                melhor_val_acc = acc_val
                paciencia_contador = 0
                caminho_checkpoint = (
                    self.dir_checkpoints / f"{nome_modelo}_epoch_{epoca:03d}.pth"
                )
                torch.save(rede.state_dict(), caminho_checkpoint)
                self.fn_log(f"  → Checkpoint salvo: {caminho_checkpoint.name}")
            else:
                paciencia_contador += 1

            # ========== EARLY STOPPING ==========
            if self.early_stop_ativo and paciencia_contador >= self.early_stop_paciencia:
                self.fn_log(
                    f"Early stopping ativado (paciência: {self.early_stop_paciencia}). "
                    f"Parando no treino."
                )
                break

        # ========== SALVAR MODELO FINAL ==========
        caminho_final = self.dir_saida / f"{nome_modelo}.pth"
        torch.save(rede.state_dict(), caminho_final)
        self.fn_log(f"Modelo final salvo: {caminho_final}")

        return {
            "accuracy": acc_treino,
            "val_accuracy": acc_val,
            "loss": loss_medio,
            "epocas_executadas": epocas_executadas,
        }
