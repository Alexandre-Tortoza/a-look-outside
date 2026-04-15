# Interface Interativa (TUI) - Galaxy Morphology Classification

Interface de usuário em texto (Text User Interface) completa e modular para configuração e execução de experimentos de classificação de morfologia galáctica.

## 🚀 Como Executar

```bash
# Modo direto
python tui_interactive.py

# Ou via main.py (futura integração)
python main.py tui
```

## 🎯 Fluxo Principal

```
┌─────────────────────────────────────────────┐
│     Tela de Boas-vindas (Menu Principal)    │
├─────────────────────────────────────────────┤
│  ▶ Novo Experimento                         │
│  📊 Listar Datasets                         │
│  🤖 Listar Modelos                          │
│  ⚙️  Ver Hardware                           │
│  ❌ Sair                                    │
└─────────────────────────────────────────────┘
          │
          ├─────────────┬──────────┬────────┬──────────┐
          │             │          │        │          │
          ▼             ▼          ▼        ▼          ▼
    ┌─────────────┐ ┌─────────┐ ┌──────┐ ┌──────────┐┌──────┐
    │  Novo Exp   │ │Datasets │ │Models│ │Hardware  ││ Sair │
    └─────────────┘ └─────────┘ └──────┘ └──────────┘└──────┘
          │
          ▼
    ┌──────────────────────┐
    │ Selecionar Dataset   │
    │  • SDSS              │
    │  • DECaLS            │
    │  • smote_sdss        │
    │  • adasyn_decals     │
    │  • [Gerar novo]      │
    └──────────────────────┘
          │
          ▼
    ┌──────────────────────┐
    │ Selecionar Modelo    │
    │  • cnn_light         │
    │  • cnn_robust        │
    │  • vit_light         │
    │  • mobilenet_robust  │
    │  • easynet_light     │
    └──────────────────────┘
          │
          ▼
    ┌──────────────────────┐
    │ Configurar Parâmetros│
    │  Épocas: [10]        │
    │  Batch Size: [32]    │
    │  Learning Rate: [0.1]│
    │  [Auto-detect HW]    │
    └──────────────────────┘
          │
          ▼
    ┌──────────────────────┐
    │ Confirmar Experimento│
    │ Resumo da config     │
    │ [Executar]           │
    └──────────────────────┘
```

## 📁 Arquitetura

```
tui/
├── __init__.py                    # Exports principais
├── app.py                         # Aplicação Textual principal
├── telas.py                       # Todas as telas interativas
└── servicos/
    ├── __init__.py
    ├── detecao_hardware.py        # CPU, GPU, RAM detection
    ├── gerenciador_datasets.py    # Descoberta e gerenciamento
    └── gerenciador_modelos.py     # Listagem de modelos

tui_interactive.py                 # Script de entrada
```

## 🔧 Serviços Disponíveis

### Detecção de Hardware (`detecao_hardware.py`)

Detecta automaticamente:
- **CPU**: Nome, cores físicos/lógicos, frequência
- **Memória**: Total, disponível, percentual de uso
- **GPU**: Modelos, memória, índice CUDA

Fornece recomendações:
```python
from tui.servicos import detectar_hardware

perfil = detectar_hardware()
print(f"Dispositivo recomendado: {perfil.dispositivo_recomendado}")
print(f"Batch Size: {perfil.batch_size_recomendado}")
print(f"DataLoader workers: {perfil.workers_recomendados}")
```

### Gerenciador de Datasets (`gerenciador_datasets.py`)

Lista e gerencia datasets:
```python
from tui.servicos import GerenciadorDatasets

gerenciador = GerenciadorDatasets()
datasets = gerenciador.listar_datasets()

# Cada dataset contém:
# - nome, tipo (bruto/balanceado), caminho
# - tamanho_mb, descricao
# - num_amostras, num_classes, tecnica_balanceamento

# Gerar novo dataset balanceado
gerenciador.gerar_balanceado(
    "Galaxy10_SDSS",
    ["smote", "undersampling"]
)
```

### Gerenciador de Modelos (`gerenciador_modelos.py`)

Descobre e lista modelos disponíveis:
```python
from tui.servicos import GerenciadorModelos

gerenciador = GerenciadorModelos()
modelos = gerenciador.listar_modelos()

# Cada modelo contém:
# - nome, classe_nome, variante, xai_metodo
# - descricao, parametros_build

# Agrupar por arquitetura
agrupados = gerenciador.agrupar_por_arquitetura()
# {'CNN': [...], 'ViT': [...], 'MobileNet': [...]}

# Resumo formatado
print(gerenciador.obter_resumo("cnn_light"))
```

## 🎮 Controles da TUI

### Navegação
- **Tab**: Navegar entre campos
- **Enter**: Confirmar seleção
- **Escape**: Voltar à tela anterior
- **Q**: Sair

### Telas Específicas
Cada tela possui seus próprios controles indicados no footer.

## 🎨 Temas Customizáveis

A TUI usa Textual com CSS customizável:

```css
/* Personalizar cores e estilos */
Screen {
    background: #1e1e2e;
    color: #e0e0e0;
}

#titulo {
    background: #7c3aed;
    text-style: bold;
}

Button {
    margin: 1;
    width: 30;
}
```

## 🔌 Extensibilidade

### Adicionar Nova Tela

```python
from textual.screen import Screen

class MinhaTelaCustomizada(Screen):
    BINDINGS = [("escape", "voltar", "Voltar")]
    
    def compose(self):
        # Implementar widgets
        pass
    
    def on_button_pressed(self, evento):
        # Tratar eventos
        pass

# Registrar na app
app.SCREENS["minha_tela"] = MinhaTelaCustomizada

# Navegar
self.app.push_screen("minha_tela")
```

### Adicionar Novo Serviço

```python
# tui/servicos/meu_servico.py
class MeuServico:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def fazer_algo(self):
        # Implementação
        pass

# Usar nas telas
from tui.servicos import MeuServico
servico = MeuServico()
```

## 🐛 Logging

Logs salvos em `tui.log`:
```
2026-04-14 19:42:02 - INFO - Iniciando Interface Interativa...
2026-04-14 19:42:03 - INFO - TUI carregada com sucesso
2026-04-14 19:42:15 - DEBUG - Dataset bruto encontrado: Galaxy10_SDSS.h5
```

## 📋 Features Implementadas

- ✅ Menu principal com navegação
- ✅ Seleção interativa de datasets
- ✅ Seleção interativa de modelos
- ✅ Configuração dinâmica de parâmetros
- ✅ Detecção automática de hardware
- ✅ Recomendações inteligentes
- ✅ Confirmação antes de executar
- ✅ Listagem de datasets
- ✅ Listagem de modelos
- ✅ Visualização de hardware
- ✅ Sistema de logging
- ✅ Navegação com atalhos

## 📋 Features Futuras

- ⚠️ Execução em background com progresso
- ⚠️ Histórico de experimentos
- ⚠️ Comparação de resultados
- ⚠️ Exportação de configuração
- ⚠️ Importação de configuração
- ⚠️ Live monitoring de treino
- ⚠️ Análise de resultados em tempo real

## 🔗 Integração com Código Existente

A TUI se integra com:

1. **Pipeline de Balanceamento** (`datasets/pipelines.py`)
   - Gera datasets balanceados sob demanda

2. **Registry de Modelos** (`models/__init__.py`)
   - Lista e instancia modelos

3. **Main.py** (futuro)
   - Executa experimentos configurados via TUI

## 🚨 Dependências

```
textual>=0.20.0
psutil>=5.8.0
torch>=1.9.0  (para detecção de GPU)
numpy>=1.20.0
scikit-learn>=1.0.0
h5py>=3.0.0
```

## 💡 Dicas de Uso

1. **Começar rápido**: Use o modo "Novo Experimento"
2. **Ver opções**: Explore "Listar Datasets" e "Listar Modelos" antes
3. **Otimizar**: Verifique as recomendações de hardware
4. **Debug**: Verifique `tui.log` se algo der errado

## 🤝 Contribuindo

Para adicionar features:

1. Criar tela nova em `tui/telas.py`
2. Criar serviço em `tui/servicos/`
3. Registrar screen em `tui/app.py`
4. Atualizar este README

## 📝 Notas

- Interface totalmente em português (pt-BR)
- Usa snake_case consistentemente
- Arquitetura desacoplada e modular
- Fácil de estender e customizar
