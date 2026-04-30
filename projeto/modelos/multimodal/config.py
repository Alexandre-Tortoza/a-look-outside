"""Hiperparâmetros do modelo Multimodal (imagem + metadados tabulares do DECaLS)."""

# O modelo multimodal só faz sentido com o dataset DECaLS que contém metadados
DATASET: str = "decals"
VERSAO_DATASET: str = "raw"
EPOCAS: int = 50
BATCH_SIZE: int = 32
LR_CABECA: float = 1e-3
LR_BACKBONE: float = 1e-4
EPOCAS_CONGELADO: int = 5
TAMANHO_IMAGEM: int = 224
SEED: int = 42
PRETRAINED: bool = True
SALVAR_PESOS: bool = True
HABILITAR_XAI: bool = False
NUM_WORKERS: int = 4
PACIENCIA_EARLY_STOP: int = 10
SCHEDULER_ATIVO: bool = True
PESO_DECAY: float = 1e-4
NOME_EXPERIMENTO: str = f"multimodal_{DATASET}_ep{EPOCAS}"

# Chaves do H5 do DECaLS que contêm metadados tabulares
# Ajustar conforme a estrutura real do arquivo galaxy10_decals.h5
FEATURES_TABULARES: list[str] = ["ra", "dec", "redshift", "mag_r", "mag_g", "mag_z"]

# Dimensões da arquitetura de fusão
DIM_VISUAL: int = 256      # dim do branch visual após GAP+Linear
DIM_TABULAR: int = 256     # dim do branch tabular após MLP
DIM_FUSAO: int = 512       # dim da camada de fusão
DROPOUT_FUSAO: float = 0.4
DROPOUT_TABULAR: float = 0.3
