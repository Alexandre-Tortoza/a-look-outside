"""Pipeline de treino do modelo Multimodal."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from dataset.carregador import CarregadorDataset
from modelos._transfer_learning import treinar_two_stage
from modelos.multimodal import config as cfg
from modelos.multimodal.modelo import MultimodalGalaxy
from modelos.treinador import HistoricoTreino
from pre_processamento.divisao_treino_teste import DivisaoDados, dividir_estratificado
from pre_processamento.normalizacao import obter_transform_avaliacao, obter_transform_treino
from utils.logger import obter_logger
from utils.reproducibilidade import fixar_semente


class DatasetMultimodal(Dataset):
    """Dataset com triplas (imagem, features_tabulares, rótulo)."""

    def __init__(
        self,
        imagens: np.ndarray,
        features: np.ndarray,
        rotulos: np.ndarray,
        transform=None,
    ) -> None:
        self.imagens = imagens
        self.features = features.astype(np.float32)
        self.rotulos = rotulos.astype(np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rotulos)

    def __getitem__(self, idx: int):
        img = self.imagens[idx]
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, torch.from_numpy(self.features[idx]), int(self.rotulos[idx])


def _criar_loaders_multimodal(
    divisao: DivisaoDados,
    features_treino: np.ndarray,
    features_val: np.ndarray,
    features_teste: np.ndarray,
    transform_treino,
    transform_val,
    batch_size: int,
    num_workers: int,
):
    """Cria DataLoaders multimodais."""
    usar_pin = torch.cuda.is_available()
    usar_pers = num_workers > 0

    def _loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=usar_pin,
                          persistent_workers=usar_pers, drop_last=shuffle)

    return (
        _loader(DatasetMultimodal(divisao.imagens_treino, features_treino, divisao.rotulos_treino, transform_treino), True),
        _loader(DatasetMultimodal(divisao.imagens_val, features_val, divisao.rotulos_val, transform_val), False),
        _loader(DatasetMultimodal(divisao.imagens_teste, features_teste, divisao.rotulos_teste, transform_val), False),
    )


def _extrair_features_tabulares(caminho_h5: Path, chaves: list[str], n: int) -> np.ndarray:
    """Extrai features tabulares do arquivo H5 do DECaLS.

    Se alguma chave não existir, substitui por zeros.
    """
    features = []
    with h5py.File(caminho_h5, "r") as f:
        for chave in chaves:
            if chave in f:
                col = f[chave][:n].astype(np.float32)
            else:
                col = np.zeros(n, dtype=np.float32)
            features.append(col.reshape(-1, 1))
    return np.concatenate(features, axis=1)  # (N, F)


def treinar(config_override: Optional[dict] = None) -> HistoricoTreino:
    params = {
        "dataset": cfg.DATASET, "epocas": cfg.EPOCAS, "batch_size": cfg.BATCH_SIZE,
        "lr_cabeca": cfg.LR_CABECA, "lr_backbone": cfg.LR_BACKBONE,
        "epocas_congelado": cfg.EPOCAS_CONGELADO, "tamanho_imagem": cfg.TAMANHO_IMAGEM,
        "seed": cfg.SEED, "pretrained": cfg.PRETRAINED, "salvar_pesos": cfg.SALVAR_PESOS,
        "num_workers": cfg.NUM_WORKERS, "paciencia_early_stop": cfg.PACIENCIA_EARLY_STOP,
        "scheduler_ativo": cfg.SCHEDULER_ATIVO, "peso_decay": cfg.PESO_DECAY,
        "nome_experimento": cfg.NOME_EXPERIMENTO, "features_tabulares": cfg.FEATURES_TABULARES,
    }
    if config_override:
        params.update(config_override)

    log = obter_logger(__name__, arquivo_log=Path("docs/multimodal/treino.log"))
    fixar_semente(params["seed"])

    carregador = CarregadorDataset()
    imagens, rotulos = carregador.carregar(params["dataset"])
    n = len(rotulos)

    # Extrair features tabulares
    caminho_h5 = carregador._resolver_caminho(params["dataset"])
    features_raw = _extrair_features_tabulares(caminho_h5, params["features_tabulares"], n)
    log.info("Features tabulares: shape=%s, chaves=%s", features_raw.shape, params["features_tabulares"])

    divisao = dividir_estratificado(imagens, rotulos, semente=params["seed"])

    # Normalizar features tabulares com StandardScaler (fit no treino)
    scaler = StandardScaler()
    idx_treino = len(divisao.rotulos_treino)
    idx_val = idx_treino + len(divisao.rotulos_val)

    # Nota: a divisão embaralha os índices; precisamos usar os mesmos índices
    # Reconstruímos os índices usando a seed para consistência
    from sklearn.model_selection import train_test_split
    indices = np.arange(n)
    fracao_treino = len(divisao.rotulos_treino) / n
    idx_tr, idx_rest = train_test_split(indices, train_size=fracao_treino,
                                        stratify=rotulos, random_state=params["seed"])
    fracao_val = len(divisao.rotulos_val) / len(idx_rest)
    idx_v, idx_te = train_test_split(idx_rest, train_size=fracao_val,
                                     stratify=rotulos[idx_rest], random_state=params["seed"])

    feat_treino = scaler.fit_transform(features_raw[idx_tr])
    feat_val = scaler.transform(features_raw[idx_v])
    feat_teste = scaler.transform(features_raw[idx_te])

    t_treino = obter_transform_treino(tamanho_imagem=params["tamanho_imagem"])
    t_val = obter_transform_avaliacao(tamanho_imagem=params["tamanho_imagem"])
    loader_treino, loader_val, _ = _criar_loaders_multimodal(
        divisao, feat_treino, feat_val, feat_teste,
        t_treino, t_val, params["batch_size"], params["num_workers"],
    )

    num_classes = len(set(rotulos.tolist()))
    num_feat = feat_treino.shape[1]
    rede = MultimodalGalaxy(
        num_features_tabulares=num_feat, pretrained=params["pretrained"]
    ).construir(num_classes=num_classes, tamanho_imagem=params["tamanho_imagem"])

    # Adaptar treinar_two_stage para aceitar batches de 3 elementos
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Usar engine própria para multimodal (batch tem 3 elementos)
    return _treinar_multimodal(
        rede=rede, nome_experimento=params["nome_experimento"],
        loader_treino=loader_treino, loader_val=loader_val,
        epocas_total=params["epocas"], epocas_congelado=params["epocas_congelado"],
        lr_cabeca=params["lr_cabeca"], lr_backbone=params["lr_backbone"],
        paciencia=params["paciencia_early_stop"], salvar_checkpoints=params["salvar_pesos"],
        scheduler_ativo=params["scheduler_ativo"], peso_decay=params["peso_decay"],
        dispositivo=dispositivo, logger=log,
    )


def _treinar_multimodal(
    rede, nome_experimento, loader_treino, loader_val,
    epocas_total, epocas_congelado, lr_cabeca, lr_backbone,
    paciencia, salvar_checkpoints, scheduler_ativo, peso_decay,
    dispositivo, logger,
) -> HistoricoTreino:
    """Engine de treino adaptada para batches (imagem, tabular, rótulo)."""
    from modelos.treinador import HistoricoTreino
    from modelos._transfer_learning import _congelar_backbone, _descongelar_tudo, _params_backbone, _params_cabeca
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from utils.visualizacao import plotar_curva_treino

    criterio = nn.CrossEntropyLoss()
    usar_amp = dispositivo.type == "cuda"
    scaler_amp = torch.cuda.amp.GradScaler() if usar_amp else None

    historico = HistoricoTreino()
    sem_melhora = 0
    caminho_pesos = Path("pesos/multimodal") / f"{nome_experimento}.pth"
    caminho_pesos.parent.mkdir(parents=True, exist_ok=True)
    rede = rede.to(dispositivo)
    scheduler = None

    for epoca in range(1, epocas_total + 1):
        if epoca == 1:
            _congelar_backbone(rede)
            otimizador = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, rede.parameters()),
                lr=lr_cabeca, weight_decay=peso_decay,
            )
        elif epoca == epocas_congelado + 1:
            _descongelar_tudo(rede)
            otimizador = torch.optim.AdamW([
                {"params": _params_backbone(rede), "lr": lr_backbone},
                {"params": _params_cabeca(rede), "lr": lr_cabeca},
            ], weight_decay=peso_decay)
            if scheduler_ativo:
                scheduler = CosineAnnealingLR(otimizador, T_max=epocas_total - epocas_congelado, eta_min=1e-6)

        if scheduler and epoca > epocas_congelado + 1:
            scheduler.step()

        # Treino
        rede.train()
        total_loss, acertos, total = 0.0, 0, 0
        for imgs, tabs, rots in loader_treino:
            imgs = imgs.to(dispositivo, non_blocking=True)
            tabs = tabs.to(dispositivo, non_blocking=True)
            rots = rots.to(dispositivo, non_blocking=True)
            otimizador.zero_grad(set_to_none=True)
            if scaler_amp:
                with torch.cuda.amp.autocast():
                    logits = rede(imgs, tabs)
                    loss = criterio(logits, rots)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(otimizador)
                nn.utils.clip_grad_norm_(rede.parameters(), 1.0)
                scaler_amp.step(otimizador)
                scaler_amp.update()
            else:
                logits = rede(imgs, tabs)
                loss = criterio(logits, rots)
                loss.backward()
                nn.utils.clip_grad_norm_(rede.parameters(), 1.0)
                otimizador.step()
            bs = rots.size(0)
            total_loss += loss.item() * bs
            acertos += (logits.argmax(1) == rots).sum().item()
            total += bs
        loss_t, acc_t = total_loss / total, acertos / total

        # Validação
        rede.eval()
        total_loss, acertos, total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, tabs, rots in loader_val:
                imgs, tabs, rots = imgs.to(dispositivo), tabs.to(dispositivo), rots.to(dispositivo)
                logits = rede(imgs, tabs)
                loss = criterio(logits, rots)
                bs = rots.size(0)
                total_loss += loss.item() * bs
                acertos += (logits.argmax(1) == rots).sum().item()
                total += bs
        loss_v, acc_v = total_loss / total, acertos / total

        historico.perdas_treino.append(loss_t)
        historico.perdas_val.append(loss_v)
        historico.accs_treino.append(acc_t)
        historico.accs_val.append(acc_v)
        historico.epocas_totais = epoca

        logger.info("Época %3d/%d | loss %.4f | acc %.4f | val_loss %.4f | val_acc %.4f",
                    epoca, epocas_total, loss_t, acc_t, loss_v, acc_v)

        if acc_v > historico.melhor_val_acc:
            historico.melhor_val_acc = acc_v
            sem_melhora = 0
            if salvar_checkpoints:
                torch.save(rede.state_dict(), caminho_pesos)
                logger.info("  ✓ Melhor modelo salvo (val_acc=%.4f)", acc_v)
        else:
            sem_melhora += 1

        if sem_melhora >= paciencia:
            logger.info("Early stopping.")
            historico.parou_cedo = True
            break

    plotar_curva_treino(
        historico.perdas_treino, historico.perdas_val,
        historico.accs_treino, historico.accs_val,
        caminho_saida=Path("docs/multimodal/curva_treino.png"),
    )
    return historico


if __name__ == "__main__":
    print(treinar().resumo())
