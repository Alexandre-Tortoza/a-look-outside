"""Parâmetros globais de pré-processamento. Nenhuma lógica aqui."""

# ---------------------------------------------------------------------------
# Tamanhos de imagem
# ---------------------------------------------------------------------------
TAMANHO_IMAGEM_SDSS: int = 69
TAMANHO_IMAGEM_DECALS: int = 256
TAMANHO_PADRAO: int = 224  # entrada para backbones pré-treinados

# ---------------------------------------------------------------------------
# Divisão treino/val/teste
# ---------------------------------------------------------------------------
FRACAO_TREINO: float = 0.70
FRACAO_VAL: float = 0.15
FRACAO_TESTE: float = 0.15

# ---------------------------------------------------------------------------
# Aumento de dados (online)
# ---------------------------------------------------------------------------
PROB_FLIP_HORIZONTAL: float = 0.5
PROB_FLIP_VERTICAL: float = 0.5
PROB_ROTACAO: float = 0.5
GRAU_ROTACAO_MAX: int = 30
BRILHO_CONTRASTE_FATOR: float = 0.2

# ---------------------------------------------------------------------------
# Balanceamento
# ---------------------------------------------------------------------------
SEMENTE_BALANCEAMENTO: int = 42
N_VIZINHOS_SMOTE: int = 5

# ---------------------------------------------------------------------------
# Normalização — stats ImageNet (fallback para fine-tuning)
# ---------------------------------------------------------------------------
MEDIA_IMAGENET: list[float] = [0.485, 0.456, 0.406]
DESVIO_IMAGENET: list[float] = [0.229, 0.224, 0.225]
