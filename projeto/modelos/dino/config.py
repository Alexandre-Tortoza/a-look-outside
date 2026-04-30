"""Hiperparâmetros do DINO (self-supervised + fine-tuning)."""

# ---------------------------------------------------------------------------
# Configurações gerais
# ---------------------------------------------------------------------------
DATASET: str = "decals"
VERSAO_DATASET: str = "raw"
TAMANHO_IMAGEM: int = 224
SEED: int = 42
NUM_WORKERS: int = 4
SALVAR_PESOS: bool = True
HABILITAR_XAI: bool = False

# ---------------------------------------------------------------------------
# Modo DINO
# "hub"        → usa dinov2 pré-treinado via torch.hub (recomendado)
# "pre_treino" → treina o DINO do zero no Galaxy10
# ---------------------------------------------------------------------------
MODO_DINO: str = "hub"
BACKBONE_DINO: str = "dinov2_vitb14"   # usado quando MODO_DINO = "hub"

# ---------------------------------------------------------------------------
# Pré-treino self-supervised (só relevante quando MODO_DINO = "pre_treino")
# ---------------------------------------------------------------------------
EPOCAS_PRE_TREINO: int = 100
BATCH_SIZE_PRE_TREINO: int = 32
LR_PRE_TREINO: float = 5e-4
WARMUP_EPOCAS: int = 10
TAMANHO_PROJECAO: int = 65536
TEMPERATURA_PROFESSOR: float = 0.04
TEMPERATURA_ALUNO: float = 0.1
BACKBONE_SCRATCH: str = "vit_small_patch16_224"  # backbone para treino do zero
NOME_EXPERIMENTO_PRE_TREINO: str = f"dino_{DATASET}_pretreino_ep{EPOCAS_PRE_TREINO}"

# ---------------------------------------------------------------------------
# Fine-tuning supervisionado
# ---------------------------------------------------------------------------
EPOCAS_AJUSTE_FINO: int = 30
BATCH_SIZE_AJUSTE_FINO: int = 32
LR_AJUSTE_FINO: float = 1e-4
LR_BACKBONE_AJUSTE: float = 1e-5
EPOCAS_CONGELADO: int = 5
PACIENCIA_EARLY_STOP: int = 10
SCHEDULER_ATIVO: bool = True
PESO_DECAY: float = 1e-4
NOME_EXPERIMENTO: str = f"dino_{DATASET}_ep{EPOCAS_AJUSTE_FINO}"
