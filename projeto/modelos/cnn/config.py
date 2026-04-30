"""Hiperparâmetros do CNN Baseline."""

DATASET: str = "decals"       # "sdss" | "decals"
VERSAO_DATASET: str = "raw"   # stem do arquivo em dataset/raw/ ou dataset/processados/
EPOCAS: int = 50
BATCH_SIZE: int = 32
LR: float = 1e-3              # CNN treinado do zero suporta LR maior
TAMANHO_IMAGEM: int = 224
SEED: int = 42
SALVAR_PESOS: bool = True
HABILITAR_XAI: bool = False
NUM_WORKERS: int = 4
PACIENCIA_EARLY_STOP: int = 10
SCHEDULER_ATIVO: bool = True
PESO_DECAY: float = 1e-4
NOME_EXPERIMENTO: str = f"cnn_{DATASET}_ep{EPOCAS}"
