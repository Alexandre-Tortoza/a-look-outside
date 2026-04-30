"""Pipeline de treino do modelo Multimodal."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from dataset.carregador import CarregadorDataset, resolver_nome_dataset
from modelos._transfer_learning import _congelar_backbone, _descongelar_tudo, _params_backbone, _params_cabeca
from modelos.multimodal.modelo import MultimodalGalaxy
from modelos.treinador import HistoricoTreino
from pre_processamento.divisao_treino_teste import DivisaoDados, dividir_estratificado
from pre_processamento.normalizacao import obter_transform_avaliacao, obter_transform_treino
from utils.checkpoint import salvar_checkpoint
from utils.config_loader import carregar_config, obter_config_modelo, obter_config_recursos
from utils.experimento import gerar_nome_experimento, registrar_run
from utils.logger import obter_logger
from utils.recursos import aplicar_batch_cap, configurar_dispositivo, obter_num_workers, usar_mixed_precision
from utils.reproducibilidade import fixar_semente
from utils.visualizacao import plotar_curva_treino


class DatasetMultimodal(Dataset):
    """Dataset com triplas (imagem, features_tabulares, rótulo)."""

    def __init__(self, imagens, features, rotulos, transform=None):
        self.imagens = imagens
        self.features = features.astype(np.float32)
        self.rotulos = rotulos.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.rotulos)

    def __getitem__(self, idx):
        img = self.imagens[idx]
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img, torch.from_numpy(self.features[idx]), int(self.rotulos[idx])


def _criar_loaders_multimodal(divisao, feat_tr, feat_val, feat_te, t_tr, t_val, bs, nw):
    usar_pin = torch.cuda.is_available()
    usar_pers = nw > 0

    def _loader(ds, shuffle):
        return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                          num_workers=nw, pin_memory=usar_pin,
                          persistent_workers=usar_pers, drop_last=shuffle)
    return (
        _loader(DatasetMultimodal(divisao.imagens_treino, feat_tr, divisao.rotulos_treino, t_tr), True),
        _loader(DatasetMultimodal(divisao.imagens_val, feat_val, divisao.rotulos_val, t_val), False),
        _loader(DatasetMultimodal(divisao.imagens_teste, feat_te, divisao.rotulos_teste, t_val), False),
    )


def _extrair_features_tabulares(caminho_h5, chaves, n):
    features = []
    with h5py.File(caminho_h5, "r") as f:
        for chave in chaves:
            if chave in f:
                col = f[chave][:n].astype(np.float32)
            else:
                col = np.zeros(n, dtype=np.float32)
            features.append(col.reshape(-1, 1))
    return np.concatenate(features, axis=1)


def treinar(config_override: Optional[dict[str, Any]] = None) -> HistoricoTreino:
    config = carregar_config()
    params = obter_config_modelo("multimodal", config)
    cfg_rec = obter_config_recursos(config)

    if config_override:
        params.update(config_override)

    dataset = params.get("dataset", "decals")
    versao = params.get("versao_dataset", "raw")
    params.setdefault("versao_dataset", versao)

    num_workers = obter_num_workers(cfg_rec)
    batch_size = aplicar_batch_cap(params.get("batch_size", 32), cfg_rec)
    params["batch_size"] = batch_size

    nome_exp = params.get("nome_experimento") or gerar_nome_experimento(
        "multimodal", dataset, versao, params["epocas"]
    )
    params["nome_experimento"] = nome_exp

    features_tab = params.get("features_tabulares", ["ra", "dec", "redshift", "mag_r", "mag_g", "mag_z"])

    log = obter_logger(__name__, arquivo_log=Path("docs/multimodal/treino.log"))
    fixar_semente(params["seed"])

    carregador = CarregadorDataset()
    nome_dataset = resolver_nome_dataset(dataset, versao)
    imagens, rotulos = carregador.carregar(nome_dataset)
    n = len(rotulos)

    caminho_h5 = carregador._resolver_caminho(nome_dataset)
    features_raw = _extrair_features_tabulares(caminho_h5, features_tab, n)
    log.info("Features tabulares: shape=%s, chaves=%s", features_raw.shape, features_tab)

    divisao = dividir_estratificado(imagens, rotulos, semente=params["seed"])

    # Normalizar features com StandardScaler (fit no treino)
    scaler = StandardScaler()
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
        divisao, feat_treino, feat_val, feat_teste, t_treino, t_val, batch_size, num_workers,
    )

    num_classes = len(set(rotulos.tolist()))
    num_feat = feat_treino.shape[1]
    rede = MultimodalGalaxy(
        num_features_tabulares=num_feat, pretrained=params.get("pretrained", True)
    ).construir(num_classes=num_classes, tamanho_imagem=params["tamanho_imagem"])

    dispositivo = configurar_dispositivo(cfg_rec)
    amp = usar_mixed_precision(cfg_rec, dispositivo)

    return _treinar_multimodal(
        rede=rede, nome_experimento=nome_exp,
        loader_treino=loader_treino, loader_val=loader_val,
        epocas_total=params["epocas"],
        epocas_congelado=params.get("epocas_congelado", 5),
        lr_cabeca=params["lr_cabeca"], lr_backbone=params["lr_backbone"],
        paciencia=params.get("paciencia_early_stop", 10),
        salvar_checkpoints=params.get("salvar_pesos", True),
        scheduler_ativo=params.get("scheduler_ativo", True),
        peso_decay=params.get("peso_decay", 1e-4),
        dispositivo=dispositivo, usar_amp=amp, params=params, logger=log,
    )


def _treinar_multimodal(
    rede, nome_experimento, loader_treino, loader_val,
    epocas_total, epocas_congelado, lr_cabeca, lr_backbone,
    paciencia, salvar_checkpoints, scheduler_ativo, peso_decay,
    dispositivo, usar_amp, params, logger,
) -> HistoricoTreino:
    """Engine de treino adaptada para batches (imagem, tabular, rótulo)."""
    from torch.optim.lr_scheduler import CosineAnnealingLR

    criterio = nn.CrossEntropyLoss()
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
                salvar_checkpoint(rede, caminho_pesos, params=params, historico=historico)
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

    registrar_run(Path("pesos"), "multimodal", nome_experimento, params, historico)

    return historico


if __name__ == "__main__":
    print(treinar().resumo())
