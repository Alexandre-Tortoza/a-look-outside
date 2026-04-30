"""Estratégias de balanceamento de classes para datasets desbalanceados."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample

from pre_processamento.config import N_VIZINHOS_SMOTE, SEMENTE_BALANCEAMENTO

# ---------------------------------------------------------------------------
# Classe base
# ---------------------------------------------------------------------------


class BalanceadorBase(ABC):
    """Contrato para todos os balanceadores."""

    def __init__(self, semente: int = SEMENTE_BALANCEAMENTO) -> None:
        self.semente = semente

    @abstractmethod
    def balancear(
        self, imagens: np.ndarray, rotulos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retorna (imagens_balanceadas, rotulos_balanceados)."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(semente={self.semente})"


# ---------------------------------------------------------------------------
# Oversampling simples (duplicação aleatória)
# ---------------------------------------------------------------------------


class BalanceadorOversampling(BalanceadorBase):
    """Oversampling por duplicação aleatória das classes minoritárias."""

    def balancear(
        self, imagens: np.ndarray, rotulos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(self.semente)
        contagens = np.bincount(rotulos)
        alvo = int(contagens.max())

        imgs_extra, rots_extra = [], []
        for classe in range(len(contagens)):
            qtd_atual = contagens[classe]
            if qtd_atual >= alvo:
                continue
            idx_classe = np.where(rotulos == classe)[0]
            qtd_faltante = alvo - qtd_atual
            idx_amostras = rng.choice(idx_classe, size=qtd_faltante, replace=True)
            imgs_extra.append(imagens[idx_amostras])
            rots_extra.append(rotulos[idx_amostras])

        if not imgs_extra:
            return imagens, rotulos

        return (
            np.concatenate([imagens] + imgs_extra, axis=0),
            np.concatenate([rotulos] + rots_extra, axis=0),
        )


# ---------------------------------------------------------------------------
# Undersampling simples (remoção aleatória)
# ---------------------------------------------------------------------------


class BalanceadorUndersampling(BalanceadorBase):
    """Undersampling por remoção aleatória das classes majoritárias."""

    def balancear(
        self, imagens: np.ndarray, rotulos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(self.semente)
        contagens = np.bincount(rotulos)
        alvo = int(contagens.min())

        imgs_sel, rots_sel = [], []
        for classe in range(len(contagens)):
            idx_classe = np.where(rotulos == classe)[0]
            idx_sel = rng.choice(idx_classe, size=alvo, replace=False)
            imgs_sel.append(imagens[idx_sel])
            rots_sel.append(rotulos[idx_sel])

        return np.concatenate(imgs_sel, axis=0), np.concatenate(rots_sel, axis=0)


# ---------------------------------------------------------------------------
# SMOTE
# ---------------------------------------------------------------------------


class BalanceadorSMOTE(BalanceadorBase):
    """SMOTE: gera amostras sintéticas por interpolação no espaço de features."""

    def __init__(self, semente: int = SEMENTE_BALANCEAMENTO, k: int = N_VIZINHOS_SMOTE) -> None:
        super().__init__(semente)
        self.k = k

    def balancear(
        self, imagens: np.ndarray, rotulos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(self.semente)
        forma_original = imagens.shape[1:]
        imgs_flat = imagens.reshape(len(imagens), -1).astype(np.float32)

        contagens = np.bincount(rotulos)
        alvo = int(contagens.max())

        imgs_sinteticas, rots_sinteticos = [], []

        for classe in range(len(contagens)):
            qtd_atual = contagens[classe]
            if qtd_atual >= alvo:
                continue

            idx_classe = np.where(rotulos == classe)[0]
            amostras = imgs_flat[idx_classe]

            k_efetivo = min(self.k, len(amostras) - 1)
            if k_efetivo < 1:
                # Apenas duplica se não há vizinhos suficientes
                qtd = alvo - qtd_atual
                idx_dup = rng.choice(idx_classe, size=qtd, replace=True)
                imgs_sinteticas.append(imgs_flat[idx_dup])
                rots_sinteticos.extend([classe] * qtd)
                continue

            nn = NearestNeighbors(n_neighbors=k_efetivo + 1)
            nn.fit(amostras)
            _, indices = nn.kneighbors(amostras)

            qtd = alvo - qtd_atual
            for _ in range(qtd):
                idx_base = rng.randint(0, len(amostras))
                idx_viz = rng.randint(1, k_efetivo + 1)
                vizinho = amostras[indices[idx_base, idx_viz]]
                alfa = rng.uniform(0, 1)
                sintetica = amostras[idx_base] + alfa * (vizinho - amostras[idx_base])
                imgs_sinteticas.append(sintetica[np.newaxis])
                rots_sinteticos.append(classe)

        if not imgs_sinteticas:
            return imagens, rotulos

        imgs_novas = np.concatenate(imgs_sinteticas, axis=0).reshape(-1, *forma_original)
        imgs_novas = imgs_novas.clip(0, 255).astype(imagens.dtype)
        return (
            np.concatenate([imagens, imgs_novas], axis=0),
            np.concatenate([rotulos, np.array(rots_sinteticos)]),
        )


# ---------------------------------------------------------------------------
# ADASYN
# ---------------------------------------------------------------------------


class BalanceadorADASYN(BalanceadorBase):
    """ADASYN: gera mais amostras sintéticas onde a densidade é menor."""

    def __init__(self, semente: int = SEMENTE_BALANCEAMENTO, k: int = N_VIZINHOS_SMOTE) -> None:
        super().__init__(semente)
        self.k = k

    def balancear(
        self, imagens: np.ndarray, rotulos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(self.semente)
        forma_original = imagens.shape[1:]
        imgs_flat = imagens.reshape(len(imagens), -1).astype(np.float32)

        contagens = np.bincount(rotulos)
        alvo = int(contagens.max())

        imgs_sinteticas, rots_sinteticos = [], []

        for classe in range(len(contagens)):
            qtd_atual = contagens[classe]
            if qtd_atual >= alvo:
                continue

            idx_classe = np.where(rotulos == classe)[0]
            amostras = imgs_flat[idx_classe]
            k_efetivo = min(self.k, len(amostras) - 1)

            if k_efetivo < 1:
                qtd = alvo - qtd_atual
                idx_dup = rng.choice(idx_classe, size=qtd, replace=True)
                imgs_sinteticas.append(imgs_flat[idx_dup])
                rots_sinteticos.extend([classe] * qtd)
                continue

            # Densidade: proporção de vizinhos de outras classes
            nn_global = NearestNeighbors(n_neighbors=k_efetivo + 1)
            nn_global.fit(imgs_flat)
            _, idx_vizinhos = nn_global.kneighbors(amostras)

            densidades = np.array([
                np.sum(rotulos[idx_vizinhos[i, 1:]] != classe) / k_efetivo
                for i in range(len(amostras))
            ], dtype=np.float32)

            soma = densidades.sum()
            if soma == 0:
                pesos = np.ones(len(amostras)) / len(amostras)
            else:
                pesos = densidades / soma

            qtd = alvo - qtd_atual
            qtds_por_amostra = (pesos * qtd).astype(int)
            qtds_por_amostra[: (qtd - qtds_por_amostra.sum())] += 1

            nn_classe = NearestNeighbors(n_neighbors=k_efetivo + 1)
            nn_classe.fit(amostras)
            _, idx_viz_classe = nn_classe.kneighbors(amostras)

            for i, n_gerar in enumerate(qtds_por_amostra):
                for _ in range(n_gerar):
                    j = rng.randint(1, k_efetivo + 1)
                    alfa = rng.uniform(0, 1)
                    sintetica = amostras[i] + alfa * (amostras[idx_viz_classe[i, j]] - amostras[i])
                    imgs_sinteticas.append(sintetica[np.newaxis])
                    rots_sinteticos.append(classe)

        if not imgs_sinteticas:
            return imagens, rotulos

        imgs_novas = np.concatenate(imgs_sinteticas, axis=0).reshape(-1, *forma_original)
        imgs_novas = imgs_novas.clip(0, 255).astype(imagens.dtype)
        return (
            np.concatenate([imagens, imgs_novas], axis=0),
            np.concatenate([rotulos, np.array(rots_sinteticos)]),
        )


# ---------------------------------------------------------------------------
# Híbrido (SMOTE + Undersampling)
# ---------------------------------------------------------------------------


class BalanceadorHibrido(BalanceadorBase):
    """Combina SMOTE (sobe minoritárias) + Undersampling (baixa majoritárias)."""

    def balancear(
        self, imagens: np.ndarray, rotulos: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        imgs_smote, rots_smote = BalanceadorSMOTE(self.semente).balancear(imagens, rotulos)
        return BalanceadorUndersampling(self.semente).balancear(imgs_smote, rots_smote)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ESTRATEGIAS: dict[str, type[BalanceadorBase]] = {
    "smote": BalanceadorSMOTE,
    "adasyn": BalanceadorADASYN,
    "oversampling": BalanceadorOversampling,
    "undersampling": BalanceadorUndersampling,
    "hibrido": BalanceadorHibrido,
}


def executar_pipeline_balanceamento(
    imagens: np.ndarray,
    rotulos: np.ndarray,
    estrategia: str,
    semente: int = SEMENTE_BALANCEAMENTO,
) -> tuple[np.ndarray, np.ndarray]:
    """Executa a estratégia de balanceamento escolhida.

    Args:
        imagens: Array (N, H, W, C).
        rotulos: Array (N,) de inteiros.
        estrategia: Uma de "smote", "adasyn", "oversampling", "undersampling", "hibrido".
        semente: Semente aleatória.

    Returns:
        (imagens_balanceadas, rotulos_balanceados).

    Raises:
        ValueError: Se a estratégia for desconhecida.
    """
    estrategia = estrategia.lower()
    if estrategia not in _ESTRATEGIAS:
        disponiveis = ", ".join(_ESTRATEGIAS.keys())
        raise ValueError(f"Estratégia '{estrategia}' inválida. Opções: {disponiveis}")

    balanceador = _ESTRATEGIAS[estrategia](semente=semente)
    return balanceador.balancear(imagens, rotulos)
