"""Hiperparâmetros do ResNet50."""

DATASET: str = "decals"
VERSAO_DATASET: str = "raw"
EPOCAS: int = 50
BATCH_SIZE: int = 32
LR_CABECA: float = 1e-3       # LR para a cabeça (camadas descongeladas)
LR_BACKBONE: float = 1e-4     # LR para o backbone (fine-tuning)
EPOCAS_CONGELADO: int = 5     # Épocas treinando só a cabeça antes de descongelar
TAMANHO_IMAGEM: int = 224
SEED: int = 42
PRETRAINED: bool = True
SALVAR_PESOS: bool = True
HABILITAR_XAI: bool = False
NUM_WORKERS: int = 4
PACIENCIA_EARLY_STOP: int = 10
SCHEDULER_ATIVO: bool = True
PESO_DECAY: float = 1e-4
NOME_EXPERIMENTO: str = f"resnet50_{DATASET}_ep{EPOCAS}"
