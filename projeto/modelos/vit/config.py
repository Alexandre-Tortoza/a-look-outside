"""Hiperparâmetros do Vision Transformer (ViT-B/16)."""

DATASET: str = "decals"
VERSAO_DATASET: str = "raw"
EPOCAS: int = 50
BATCH_SIZE: int = 32
LR_CABECA: float = 1e-3
LR_BACKBONE: float = 1e-5     # ViT precisa de LR bem menor no backbone
EPOCAS_CONGELADO: int = 10    # Mais épocas aquecendo a cabeça antes de descongelar
TAMANHO_IMAGEM: int = 224     # Obrigatório para ViT-B/16 pré-treinado
SEED: int = 42
PRETRAINED: bool = True
BACKBONE: str = "vit_base_patch16_224"
SALVAR_PESOS: bool = True
HABILITAR_XAI: bool = False
NUM_WORKERS: int = 4
PACIENCIA_EARLY_STOP: int = 10
SCHEDULER_ATIVO: bool = True
PESO_DECAY: float = 1e-4
NOME_EXPERIMENTO: str = f"vit_{DATASET}_ep{EPOCAS}"
