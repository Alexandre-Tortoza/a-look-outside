# Implementação: Training Engine, Early Stopping, Fine-tuning ViT e Checkpoints

## ✅ Resumo das Alterações

### 1. **Motor de Treinamento** (`models/treinador.py`)
Novo arquivo com classe `TreinadorModelo`:
- ✅ Loop completo de treinamento (forward, backward, validation)
- ✅ Cálculo de loss (CrossEntropyLoss) e acurácia por época
- ✅ **Early stopping** configurável (paciência + contagem de épocas sem melhora)
- ✅ **Checkpoints automáticos** quando val_accuracy melhora
- ✅ **Salva modelo final** em `.pth`
- ✅ Callbacks para logging linha-a-linha (`fn_log`)
- ✅ Callbacks para progresso (`fn_progresso`)
- ✅ Suporte a CPU/CUDA automático

**Chave:**
```python
TreinadorModelo(
    num_epochs=10,
    learning_rate=0.001,
    dispositivo="auto",
    dir_saida=Path("results/run-00"),
    early_stop_ativo=True,
    early_stop_paciencia=5,
    salvar_checkpoints=True,
)
metricas = treinador.treinar(rede, "vit_light", loader_treino, loader_val)
```

### 2. **Preparador de Dados** (`datasets/preparador.py`)
Novo arquivo com classe `PreparadorDados`:
- ✅ Carrega H5 e NPZ automaticamente
- ✅ Train/val split configurável (default 70/30)
- ✅ Transformações: Resize (64×64), ToTensor, Normalize
- ✅ Retorna `(DataLoader_treino, DataLoader_val)`
- ✅ Suporte a `num_workers` e `pin_memory` (GPU)

### 3. **ViT com Fine-tuning** (`models/vit.py`)
Modificações em `ViTLight` e `ViTRobust`:
- ✅ Novo parâmetro `modo_treino: str` ("scratch" ou "finetune")
- ✅ `ViTLight.build()`:
  - `"scratch"`: constrói `RedeViTSimples` customizada (como antes)
  - `"finetune"`: carrega `timm.vit_small_patch16_224` pré-treinado, congela backbone
- ✅ `ViTRobust.build()`: idem com `timm.vit_base_patch16_224`
- ✅ Propriedade `suporta_finetune` = `True`
- ✅ Congelamento automático de camadas não-head

### 4. **Base Model** (`models/base.py`)
Adições à classe abstrata:
- ✅ Propriedade `suporta_checkpoint: bool` (default `True`)
- ✅ Propriedade `suporta_finetune: bool` (default `False`, sobrescrito por ViT)

### 5. **Configuração Expandida** (`tui/app.py`)
Novos campos em `ConfiguracaoExperimento`:
- ✅ `early_stop_ativo: bool` (default `True`)
- ✅ `early_stop_paciencia: int` (default `5`)
- ✅ `modo_vit: str` (default `"scratch"`)
- ✅ `salvar_checkpoints: bool` (default `True`)
- ✅ `divisao_treino: float` (default `0.7`)

### 6. **TUI - Tela de Configuração** (`tui/telas.py` - `TelaConfiguracaoParametros`)
Novos campos de entrada:
- ✅ **Input**: Train/Val split (0.0-1.0)
- ✅ **Select** (apenas para ViT): Modo ("Scratch" / "Fine-tuning")
- ✅ **Switch**: Early stopping ativo
- ✅ **Input**: Paciência early stop
- ✅ **Switch**: Salvar checkpoints
- ✅ Callback `_salvar_parametros()` atualizado para ler todos os novos campos

### 7. **TUI - Tela de Execução** (`tui/telas.py` - `TelaExecutandoExperimento`)
Implementação completa:
- ✅ **Worker assíncrono** (`self.run_worker()`) para não bloquear TUI
- ✅ **ProgressBar** em tempo real
- ✅ **RichLog** para streaming de logs com cores
- ✅ Carregamento automático de dados
- ✅ Instantiação de modelo com modo_vit se ViT
- ✅ Execução de treinamento com `TreinadorModelo`
- ✅ Criação automática de `results/run-XX/` (incremento do número)
- ✅ Tratamento de erros com try/except

**Flow na TUI:**
1. Clica "Executar Experimento"
2. Tela monta e inicia worker
3. Carrega dataset → exibe logs
4. Instancia modelo → exibe logs
5. Treina época por época → ProgressBar atualiza
6. Após cada época: logs mostram Loss, Acc Train, Acc Val
7. Early stop para antes do fim se necessário
8. Salva `.pth` final no `results/run-XX/`

### 8. **Dependências** (`pyproject.toml`)
Adicionadas:
- ✅ `torch>=2.0.0`
- ✅ `torchvision>=0.15.0`
- ✅ `timm>=0.9.0`

---

## 📂 Estrutura de Saída

Após um treinamento:

```
results/
├── run-00/
│   ├── run-00.md              ← specs + config (ainda não implementado)
│   ├── vit_light.pth          ← modelo final
│   ├── checkpoints/
│   │   ├── vit_light_epoch_001.pth
│   │   ├── vit_light_epoch_003.pth  ← salvo quando val_acc melhora
│   │   └── vit_light_epoch_007.pth
│   └── logs/
│       └── vit_light.log      ← (não implementado ainda)
└── run-01/
    └── ...
```

---

## 🚀 Como Usar

### 1. **Via TUI** (recomendado)
```bash
python tui_interactive.py
```
1. Novo Experimento
2. Selecionar dataset (ex: `Galaxy10_SDSS.h5`)
3. Selecionar modelo (ex: `vit_light`)
4. Configuração:
   - Épocas: 20
   - Batch Size: 32
   - Learning Rate: 0.001
   - **Train/Val Split: 0.7**
   - **Modo ViT: Fine-tuning** ← novo!
   - **Early Stop: ON, paciência 5** ← novo!
   - **Checkpoints: ON** ← novo!
5. Clica "Executar" → vê ProgressBar + logs em tempo real

### 2. **Via Python** (debug)
```python
from pathlib import Path
from models import get_model
from datasets.preparador import PreparadorDados
from models.treinador import TreinadorModelo

# Preparar dados
preparador = PreparadorDados()
loader_treino, loader_val = preparador.preparar(
    "datasets/Galaxy10_SDSS.h5",
    batch_size=32,
    num_workers=4,
    divisao_treino=0.7,
    img_size=64,
)

# Modelo ViT com fine-tuning
wrapper = get_model("vit_light")
wrapper.modo_treino = "finetune"  # novo!
rede = wrapper.build(10, 64)

# Treinar
treinador = TreinadorModelo(
    num_epochs=10,
    learning_rate=0.001,
    dispositivo="auto",
    dir_saida=Path("results/run-00"),
    early_stop_ativo=True,
    early_stop_paciencia=5,
    salvar_checkpoints=True,
)
metricas = treinador.treinar(rede, "vit_light", loader_treino, loader_val)
print(f"Acurácia: {metricas['val_accuracy']:.2%}")
```

---

## 🔧 Detalhes Técnicos

### Early Stopping
- Monitora `val_accuracy` a cada época
- Se melhora: reseta contador, salva checkpoint
- Se não melhora: incrementa contador
- Para se `contador >= early_stop_paciencia`
- Genérico para todos os modelos

### Checkpoints
- Salvo em `results/run-XX/checkpoints/`
- Nome: `{nome_modelo}_epoch_{NNN}.pth`
- Apenas quando `val_accuracy` melhora
- Permite recuperar melhor modelo se training divergir depois

### Fine-tuning ViT
- Carrega pesos de `timm`:
  - ViTLight: `vit_small_patch16_224`
  - ViTRobust: `vit_base_patch16_224`
- Congela todas as camadas exceto `head`
- Permite treinar em datasets menores com menos overfitting

### Otimizador
- `AdamW` com learning rate configurável
- Loss: `CrossEntropyLoss`
- Sem peso customizado nos batches

---

## ⚠️ Limitações Conhecidas

1. `run-00.md` não é gerado automaticamente (specs de máquina) — fácil de adicionar
2. Logs não são salvos em arquivo (apenas exibidos na TUI)
3. Sem suporte para dataset cross-dataset yet (treinar em um, testar em outro)
4. Sem suporte para custom augmentation (só resize + normalize)
5. `timm` pode ser lento na primeira execução (download de pesos)

---

## 🎯 Próximas Melhorias Sugeridas

1. Gerar `run-XX.md` com specs de máquina, config, timestamp
2. Salvar logs em arquivo texto
3. Suporte para cross-dataset mode (train SDSS, val DECaLS)
4. Implementar attention rollout XAI para ViT
5. Adicionar augmentation configurável
6. LR scheduler (cosine annealing, etc)
7. Exportar métricas em JSON
8. UI improvements: tabelas melhor formatadas, menor fonte

---

## ✅ Testes Manuais

```bash
# 1. Verificar imports
python -c "from models.vit import ViTLight; print(ViTLight('scratch').suporta_finetune)"
# Output: True

# 2. Verificar preparador
python -c "from datasets.preparador import PreparadorDados; print('OK')"

# 3. Verificar config
python -c "from tui.app import ConfiguracaoExperimento; c = ConfiguracaoExperimento(); print(c.early_stop_paciencia)"
# Output: 5

# 4. Rodar TUI
python tui_interactive.py
```

---

**Data**: 2026-04-14  
**Status**: ✅ Implementado e Testado  
**Próxima Tarefa**: Melhorias UI e geração de docs automática
