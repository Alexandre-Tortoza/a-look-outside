# Arquitetura dos Modelos de Classificação Galáctica

## Estrutura de Módulo (Clean Code)

Cada arquivo segue o padrão:

```
models/
├── easynet.py       # RedeEasyNetSimples, RedeEasyNetRobusta, EasyNet, EasyNetRobust
├── cnn.py           # RedeCNNSimples, RedeCNNRobusta, CNNLight, CNNRobust
├── mobilenet.py     # RedeMovelSimples, RedeMovelRobusta, MobileNetLight, MobileNetRobust
├── vit.py           # RedeViTSimples, RedeViTRobusta, ViTLight, ViTRobust
├── base.py          # GalaxyClassifier (classe abstrata)
└── __init__.py      # Registry de modelos
```

---

## Padrão de Design: Wrapper + Model

### Camada 1: Modelo de Rede (nn.Module)
Implementação real da arquitetura neural. Totalmente independente, testável isoladamente.

```python
class RedeEasyNetSimples(nn.Module):
    """Rede PyTorch real, sem dependência do benchmark."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = ...
    def forward(self, x):
        return self.classificador(self.features(x))

# Uso direto:
rede = RedeEasyNetSimples(10)
saida = rede(torch.randn(2, 3, 64, 64))  # ✓ Funciona!
```

### Camada 2: Wrapper (GalaxyClassifier)
Adapter para o benchmark. Responsável por orquestração, metadata, e XAI.

```python
class EasyNet(GalaxyClassifier):
    """Wrapper que implementa a interface do benchmark."""
    def build(self, num_classes, img_size):
        return RedeEasyNetSimples(num_classes)
    
    def explain(self, model, input_tensor, target_class=None):
        # Lógica de explicabilidade
        ...

# Uso via benchmark:
m = get_model("easynet_light")
rede = m.build(num_classes=10, img_size=64)
```

**Benefício:** Separação clara de responsabilidades.

---

## Nomes em PT-BR (Clean Code)

### Convenção Adotada

| Tipo | Padrão | Exemplo | Explicação |
|---|---|---|---|
| **Classe de rede** | `Rede{Modelo}{Variante}` | `RedeEasyNetSimples`, `RedeCNNRobusta` | Substantivo + adjetivo |
| **Wrapper** | `{Modelo}{Variante}` | `EasyNet`, `CNNRobust` | Mantém compatibilidade com registry |
| **Componente privado** | `_{Descricao}` | `_BlocoResidual`, `_BlocoTransformador` | Prefixo `_` indica uso interno |
| **Variável** | `snake_case` | `tamanho_imagem`, `dimensao_embed` | Padrão Python |

### Por Que PT-BR?

- ✓ Código mais legível para pesquisadores/academicistas da região
- ✓ Docstrings em pt-br justificam nomes em pt-br
- ✓ Clean Code: nomes descritivos > nomes originais obscuros
- ✗ NÃO usar pt-br se sacrificar clareza (ex: `ConversorCaracteristicaNumericaProfunda` em vez de `FeatureEncoder`)

---

## Componentes Reutilizáveis (DRY)

### _BlocoResidual (CNN)
```python
# cnn.py
class _BlocoResidual(nn.Module):
    """Bloco residual com skip connection."""
```
Usado em: **RedeCNNRobusta**

### _ConvSeparavelProfundidade (MobileNet)
```python
# mobilenet.py
class _ConvSeparavelProfundidade(nn.Module):
    """Depthwise + Pointwise conv."""
```
Usado em: **RedeMovelSimples**

### _BlocoResidualInvertido (MobileNet)
```python
# mobilenet.py
class _BlocoResidualInvertido(nn.Module):
    """MobileNetV2-style inverted residual."""
```
Usado em: **RedeMovelRobusta**

### _IncorporadorPatch (ViT)
```python
# vit.py
class _IncorporadorPatch(nn.Module):
    """Converte imagem em patches embedados."""
```
Usado em: **RedeViTSimples**, **RedeViTRobusta**

### _BlocoTransformador (ViT)
```python
# vit.py
class _BlocoTransformador(nn.Module):
    """Bloco transformer com atenção multi-cabeça e MLP."""
```
Usado em: **RedeViTSimples**, **RedeViTRobusta**

**Vantagem:** Sem duplicação. Mudanças em um lugar afetam todos os usuários.

---

## Testabilidade

### Antes (Anti-padrão)
```python
# ❌ Classes aninhadas em métodos
class EasyNet(GalaxyClassifier):
    def build(self, num_classes, img_size):
        class _EasyNet(nn.Module):  # Onde testar isso?
            ...
        return _EasyNet(num_classes)

# Impossível:
# from models.easynet import _EasyNet
```

### Depois (Clean Code)
```python
# ✓ Classes no escopo do módulo
class RedeEasyNetSimples(nn.Module):
    ...

# Possível:
from models.easynet import RedeEasyNetSimples

# Teste unitário direto:
def test_rede_easynet_simples():
    rede = RedeEasyNetSimples(10)
    x = torch.randn(2, 3, 64, 64)
    y = rede(x)
    assert y.shape == (2, 10)
```

---

## Adicionar Novo Modelo: Checklist

Para adicionar uma nova arquitetura (ex: `ResNet`):

- [ ] Criar `resnet.py`
- [ ] Implementar `RedeResNetSimples(nn.Module)` com `build()` real
- [ ] Implementar `RedeResNetRobusta(nn.Module)` com `build()` real
- [ ] Extrair componentes reutilizáveis com prefixo `_` (ex: `_BlocoResNetBottleneck`)
- [ ] Implementar wrappers `ResNetLight(GalaxyClassifier)` e `ResNetRobust`
- [ ] Adicionar ao registry em `__init__.py`:
  ```python
  from models.resnet import ResNetLight, ResNetRobust
  def get_model(name):
      models = {
          ...
          "resnet_light": ResNetLight,
          "resnet_robust": ResNetRobust,
      }
  ```
- [ ] Testar: `python3 -c "from models import get_model; m = get_model('resnet_light'); ..."`

---

## Verificação de Qualidade

### Verificar desacoplamento
```bash
# Nenhuma classe aninhada em métodos?
grep -r "class _" models/*.py | grep -v "^[^:]*: " | wc -l  # Deve ser 0 (no matches)
```

### Verificar importabilidade
```bash
python3 -c "
from models.cnn import RedeCNNSimples, _BlocoResidual
from models.vit import _BlocoTransformador
print('✓ Todos os componentes são importáveis')
"
```

### Verificar registry
```bash
python3 -c "
from models import list_models, get_model
print('Modelos registrados:', list_models())
for m in list_models():
    get_model(m)
print('✓ Todos os modelos podem ser instantiados')
"
```

---

## Referências

- **Clean Code (Robert C. Martin)**: Cap 2 (Nomes significativos), Cap 3 (Funções)
- **PEP 8**: Convenção de nomes Python
- **SOLID**: Single Responsibility Principle
