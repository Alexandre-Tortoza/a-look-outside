"""Hiperparâmetros do VGG16."""

DATASET: str = "decals"
VERSAO_DATASET: str = "raw"
EPOCAS: int = 50
BATCH_SIZE: int = 16          # VGG16 é pesado; batch menor
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
NOME_EXPERIMENTO: str = f"vgg16_{DATASET}_ep{EPOCAS}"
