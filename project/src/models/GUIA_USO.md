# Guia Rápido: Como Usar os Modelos Refatorizados

## 🚀 Uso Rápido

### Opção 1: Via Benchmark (Como Antes)
```python
from models import get_model
import torch

# Obter modelo via registry
classificador = get_model("cnn_light")

# Instanciar a rede
rede = classificador.build(num_classes=10, img_size=64)

# Usar
entrada = torch.randn(2, 3, 64, 64)
saida = rede(entrada)  # Shape: [2, 10]
```

### Opção 2: Uso Direto de Rede (Novo!)
```python
from models.cnn import RedeCNNSimples
import torch

# Instanciar a rede diretamente
rede = RedeCNNSimples(num_classes=10)

# Usar (sem wrapper)
entrada = torch.randn(2, 3, 64, 64)
saida = rede(entrada)  # Shape: [2, 10]
```

### Opção 3: Teste de Componente (Novo!)
```python
from models.cnn import _BlocoResidual
import torch

# Testar componente isolado
bloco = _BlocoResidual(in_ch=64, out_ch=128)
entrada = torch.randn(2, 64, 32, 32)
saida = bloco(entrada)  # Shape: [2, 128, 32, 32]
```

---

## 📦 Importações Disponíveis

### EasyNet
```python
from models.easynet import (
    RedeEasyNetSimples,      # nn.Module
    RedeEasyNetRobusta,      # nn.Module
    EasyNet,                 # GalaxyClassifier
    EasyNetRobust,           # GalaxyClassifier
)
```

### CNN
```python
from models.cnn import (
    _BlocoResidual,          # Componente privado (reutilizável)
    RedeCNNSimples,          # nn.Module
    RedeCNNRobusta,          # nn.Module
    CNNLight,                # GalaxyClassifier
    CNNRobust,               # GalaxyClassifier
)
```

### MobileNet
```python
from models.mobilenet import (
    _ConvSeparavelProfundidade,      # Componente privado
    _BlocoResidualInvertido,         # Componente privado
    RedeMovelSimples,                # nn.Module
    RedeMovelRobusta,                # nn.Module
    MobileNetLight,                  # GalaxyClassifier
    MobileNetRobust,                 # GalaxyClassifier
)
```

### ViT
```python
from models.vit import (
    _IncorporadorPatch,      # Componente privado (compartilhado)
    _BlocoTransformador,     # Componente privado (compartilhado)
    RedeViTSimples,          # nn.Module
    RedeViTRobusta,          # nn.Module
    ViTLight,                # GalaxyClassifier
    ViTRobust,               # GalaxyClassifier
)
```

---

## 🧪 Exemplos de Teste

### Teste Unitário de Rede
```python
import pytest
import torch
from models.cnn import RedeCNNSimples

def test_rede_cnn_simples_output_shape():
    """Verifica que output tem shape correto."""
    rede = RedeCNNSimples(num_classes=10)
    entrada = torch.randn(2, 3, 64, 64)
    saida = rede(entrada)
    
    assert saida.shape == (2, 10)

def test_rede_cnn_simples_with_diferent_sizes():
    """Testa com batch sizes diferentes."""
    rede = RedeCNNSimples(num_classes=10)
    
    for batch_size in [1, 4, 8, 16]:
        entrada = torch.randn(batch_size, 3, 64, 64)
        saida = rede(entrada)
        assert saida.shape == (batch_size, 10)
```

### Teste de Componente
```python
import torch
from models.cnn import _BlocoResidual

def test_bloco_residual():
    """Verifica bloco residual."""
    bloco = _BlocoResidual(in_ch=64, out_ch=128)
    entrada = torch.randn(2, 64, 32, 32)
    saida = bloco(entrada)
    
    assert saida.shape == (2, 128, 32, 32)
```

### Teste via Registry (Compatibilidade)
```python
import torch
from models import get_model

def test_todos_modelos():
    """Verifica que todos os modelos funcionam."""
    modelos = [
        'easynet_light', 'easynet_robust',
        'cnn_light', 'cnn_robust',
        'mobilenet_light', 'mobilenet_robust',
        'vit_light', 'vit_robust'
    ]
    
    for nome_modelo in modelos:
        classificador = get_model(nome_modelo)
        rede = classificador.build(num_classes=10, img_size=64)
        
        entrada = torch.randn(2, 3, 64, 64)
        saida = rede(entrada)
        
        assert saida.shape == (2, 10), f"Falha em {nome_modelo}"
```

---

## 🔍 Diferenças Entre Abordagens

| Aspecto | Via Registry | Via Importação Direta |
|---|---|---|
| **Interface** | `get_model("cnn_light")` | `RedeCNNSimples()` |
| **Responsabilidade** | Orquestração completa (XAI, metadata) | Apenas inferência |
| **Testabilidade** | Testa wrapper + modelo | Testa modelo isolado |
| **Flexibilidade** | Menos (preso ao benchmark) | Mais (pode usar em qualquer lugar) |
| **Recomendado para** | Pipeline do benchmark | Pesquisa/experimentos/componentes |

---

## 💡 Casos de Uso

### 1. Executar o Benchmark Completo
```bash
cd project/src
python main.py --models cnn_light cnn_robust --dataset decals
```
→ Usa wrappers (GalaxyClassifier)

### 2. Treinar Um Modelo Específico
```python
from models.cnn import RedeCNNSimples
import torch.optim as optim

rede = RedeCNNSimples(num_classes=10)
otimizador = optim.Adam(rede.parameters())

# Seu loop de treinamento aqui
for entrada, alvo in dataloader:
    saida = rede(entrada)
    perda = criterio(saida, alvo)
    # ...
```

### 3. Experimentar com Componentes
```python
from models.vit import _BlocoTransformador

# Configurar diferente:
bloco = _BlocoTransformador(dimensao_embed=256, num_cabecas=8)

# Usar em novo ViT customizado:
class MeuViTCustomizado(nn.Module):
    def __init__(self):
        self.bloco1 = _BlocoTransformador(256, 8)
        self.bloco2 = _BlocoTransformador(256, 8)
```

### 4. Comparar Arquiteturas
```python
from models.easynet import RedeEasyNetSimples
from models.cnn import RedeCNNSimples
from models.vit import RedeViTSimples

redes = [
    RedeEasyNetSimples(10),
    RedeCNNSimples(10),
    RedeViTSimples(64, 16, 192, 4, 6, 10),
]

for rede in redes:
    print(f"{rede.__class__.__name__}")
    print(f"  Parâmetros: {sum(p.numel() for p in rede.parameters())}")
```

---

## ⚠️ Notas Importantes

### ViT e Tamanho de Imagem
```python
# ✅ Funciona (64 é divisível por 16 e 8)
rede = RedeViTSimples(64, 16, 192, 4, 6, 10)

# ❌ Quebra (69 não é divisível por 16)
# AssertionError: img_size=69 deve ser divisível por patch_size=16
rede = RedeViTSimples(69, 16, 192, 4, 6, 10)

# Solução: redimensione a imagem para 64px
```

### Device (CPU/GPU)
```python
import torch

rede = RedeCNNSimples(10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rede = rede.to(device)
entrada = torch.randn(2, 3, 64, 64).to(device)
saida = rede(entrada)
```

---

## 📚 Documentação Adicional

Veja também:
- `README_ARQUITETURA.md` — Design patterns, componentes reutilizáveis
- `../../../.claude/plans/resumo-refactoring.md` — Comparação antes/depois
- `../../../.claude/projects/.../memory/clean_code_pytorch.md` — Padrões usados
