# 🌌 Guia Completo da TUI Interativa

## Visão Geral

A TUI (Text User Interface) é uma interface moderna e interativa em linha de comando que permite configurar e executar experimentos de classificação de morfologia galáctica com máximo controle e facilidade.

## 📦 O Que Foi Implementado

### 1. **Arquitetura Modular e Desacoplada**

```
tui/
├── __init__.py                  # Exports públicos
├── app.py                       # Aplicação Textual principal
├── telas.py                     # 7 telas interativas
└── servicos/
    ├── __init__.py
    ├── detecao_hardware.py      # Detecta CPU, GPU, RAM
    ├── gerenciador_datasets.py  # Gerencia datasets
    └── gerenciador_modelos.py   # Gerencia modelos
```

**Padrão de Design:**
- Camada de **Apresentação**: `app.py` + `telas.py`
- Camada de **Lógica**: `servicos/`
- Camada de **Integração**: Acesso a `datasets/`, `models/`

### 2. **Seleção Dinâmica de Datasets**

- **Descobre automaticamente** datasets brutos (H5) e balanceados (NPZ)
- **Permite gerar novos** datasets balanceados sob demanda
- **Exibe informações**: tamanho, tipo, técnica de balanceamento

```python
GerenciadorDatasets().listar_datasets()
# Retorna: [InfoDataset(...), InfoDataset(...), ...]
```

### 3. **Seleção Dinâmica de Modelos**

- **Descobre modelos** a partir do registry
- **Agrupa por arquitetura**: CNN, ViT, MobileNet, EasyNet
- **Extrai parâmetros** dinamicamente do método `build()`

```python
GerenciadorModelos().listar_modelos()
# Retorna: [InfoModelo(nome='cnn_light', variante='light', ...), ...]
```

### 4. **Detecção Automática de Hardware**

Detecta e recomenda:

```python
perfil = detectar_hardware()

# CPU
perfil.cpu.nome              # "Intel Core i7-9700K"
perfil.cpu.nucleos_fisicos   # 8
perfil.cpu.nucleos_logicos   # 8
perfil.cpu.frequencia_mhz    # 3600.0

# Memória
perfil.memoria.total_gb      # 32.0
perfil.memoria.disponivel_gb # 24.5
perfil.memoria.percentual_uso # 23.4

# GPU
perfil.gpus                  # [InfoGPU(...), ...]
perfil.tem_gpu               # True
perfil.dispositivo_recomendado  # "cuda:0"

# Recomendações
perfil.batch_size_recomendado   # 64
perfil.workers_recomendados     # 8
```

### 5. **Configuração Dinâmica de Parâmetros**

- **Parâmetros adaptáveis** por modelo
- **Valores padrão** extraídos da arquitetura
- **Validação automática** de tipos
- **Sugestões inteligentes** baseadas em hardware

Parâmetros configuráveis:
- Épocas
- Batch Size
- Learning Rate
- Dispositivo (CPU/GPU)
- Workers para DataLoader

### 6. **Fluxo Guiado**

```
Menu Principal
    ├─ [▶] Novo Experimento
    │   ├─ Selecionar Dataset
    │   ├─ Selecionar Modelo
    │   ├─ Configurar Parâmetros
    │   ├─ Confirmar Experimento
    │   └─ Executar
    │
    ├─ [📊] Listar Datasets
    │   └─ Exibir informações de todos os datasets
    │
    ├─ [🤖] Listar Modelos
    │   └─ Exibir arquiteturas e variantes
    │
    ├─ [⚙️] Hardware
    │   └─ Mostrar especificações e recomendações
    │
    └─ [❌] Sair
```

### 7. **Extensibilidade**

Fácil adicionar:

**Novo Dataset:**
- Coloque arquivo H5 ou NPZ em `datasets/`
- `GerenciadorDatasets` descobre automaticamente

**Novo Modelo:**
- Adicione classe herdando de `GalaxyClassifier` em `models/`
- Registre em `models/__init__.py`
- `GerenciadorModelos` descobre automaticamente

**Nova Tela:**
```python
class MinhaTelaCustomizada(TextualScreen):
    BINDINGS = [("escape", "voltar", "Voltar")]
    
    def compose(self):
        # Widgets Textual
        yield Header()
        yield Container(...)
        yield Footer()
    
    def action_voltar(self):
        self.app.pop_screen()

# Registrar
app.SCREENS["minha_tela"] = MinhaTelaCustomizada
```

## 🚀 Como Executar

### Opção 1: Script Direto

```bash
python tui_interactive.py
```

### Opção 2: Via CLI Customizada (Futuro)

```bash
python main.py tui
```

### Opção 3: Programaticamente

```python
from tui import executar

executar()
```

## 📋 Telas Implementadas

### 1️⃣ **Tela de Boas-vindas**
- Menu principal com opções
- Acesso rápido a todas as funções
- Logout/Sair

### 2️⃣ **Seleção de Dataset**
- Lista datasets disponíveis
- Exibe tipo, tamanho, técnica
- Opção para gerar novo dataset

### 3️⃣ **Seleção de Modelo**
- Lista modelos por arquitetura
- Exibe variante e método XAI
- Agrupamento inteligente

### 4️⃣ **Configuração de Parâmetros**
- Entrada interativa de valores
- Hardware auto-detectado
- Recomendações em tempo real

### 5️⃣ **Confirmação de Experimento**
- Resumo de todas as escolhas
- Botão para executar
- Opção de voltar para ajustar

### 6️⃣ **Listagem de Datasets**
- Visão geral de todos os datasets
- Informações detalhadas
- Sem selecionar

### 7️⃣ **Listagem de Modelos**
- Visão geral de todos os modelos
- Parâmetros e arquitetura
- Sem selecionar

### 8️⃣ **Informações de Hardware**
- Especificações detalhadas
- CPU, memória, GPU
- Recomendações otimizadas

## 🎯 Casos de Uso

### Cenário 1: Iniciante
```
1. Abre a TUI
2. Clica em "Novo Experimento"
3. Seleciona SDSS (dataset padrão)
4. Seleciona CNN Light (modelo simples)
5. Usa valores padrão
6. Confirma e executa
```

### Cenário 2: Pesquisador Experiente
```
1. Abre a TUI
2. Clica em "Listar Datasets"
3. Verifica quais datasets foram gerados
4. Volta e inicia novo experimento
5. Seleciona dataset_smote (balanceado)
6. Seleciona ViT Robust (modelo complexo)
7. Ajusta parâmetros: epochs=50, batch_size=64
8. Confirma e executa
```

### Cenário 3: Benchmarking
```
1. Listar todos os modelos
2. Listar todos os datasets
3. Configurar 8 experimentos diferentes
4. Executar em sequência ou paralelo
5. Comparar resultados
```

## 🔧 Configurações Recomendadas

### Para GPU (NVIDIA)

```
Dispositivo: cuda:0
Batch Size: 128-256
Workers: 4-8
Learning Rate: 0.001-0.01
Épocas: 20-50
```

### Para CPU

```
Dispositivo: cpu
Batch Size: 32-64
Workers: 4 (máximo)
Learning Rate: 0.001
Épocas: 50-100
```

### Para RAM Limitado (< 8GB)

```
Dispositivo: cpu
Batch Size: 8-16
Workers: 0-2
Learning Rate: 0.001
Épocas: 10-20
```

## 📊 Exemplo de Output

```
✓ Hardware detectado: 10 cores
✓ Datasets encontrados: 3
  - Galaxy10_SDSS (bruto)
  - Galaxy10_DECals (bruto)
  - sdss_undersampling (balanceado)

✓ Modelos encontrados: 8
  - cnn_light (light)
  - cnn_robust (robust)
  - vit_light (light)
  - vit_robust (robust)
  - mobilenet_light (light)
  - mobilenet_robust (robust)
  - easynet_light (light)
  - easynet_robust (robust)

✓ Hardware recomendado:
  - Dispositivo: cuda:0
  - Batch Size: 64
  - Workers: 4
```

## 🎨 Personalização

### Alterar Tema

Editar em `tui/app.py`:

```python
CSS = """
Screen {
    background: $surface;
    color: $text;
}

#titulo {
    background: $accent;
    color: $text;
}

Button {
    margin: 1;
    width: 30;
}
"""
```

### Mudar Cores

```python
def get_css_variables(self) -> dict:
    return {
        "surface": "#1a1a2e",      # Preto azulado
        "accent": "#16c784",        # Verde
        "text": "#ffffff",          # Branco
    }
```

## 🐛 Troubleshooting

### Erro: "Module not found: textual"

```bash
pip install textual psutil
```

### Erro: "Module not found: sklearn"

```bash
pip install scikit-learn
```

### TUI não inicia

```bash
# Verificar logs
cat tui.log

# Executar com debug
python -u tui_interactive.py
```

### Hardware não detectado

```bash
# Checar psutil
python -c "import psutil; print(psutil.cpu_count())"

# Checar torch para GPU
python -c "import torch; print(torch.cuda.is_available())"
```

## 📈 Próximas Implementações

- ⏳ Execução em background com barra de progresso
- 📊 Dashboard de monitoramento em tempo real
- 💾 Salvamento/carregamento de configurações
- 📈 Comparação de experimentos
- 🔄 Histórico de execuções
- 🎯 Validação cruzada automática
- 📸 Screenshot de resultados

## 🔗 Integração com Código Existente

A TUI se conecta com:

```python
# 1. Pipeline de Balanceamento
from datasets import PipelineBalanceamento
pipeline = PipelineBalanceamento()

# 2. Modelos
from models import get_model, list_models
modelo = get_model("cnn_light")

# 3. Carregador de Datasets
from datasets import CarregadorDataset
carregador = CarregadorDataset()
```

## 📝 Código Limpo

- ✅ **PEP 8**: Segue padrões Python
- ✅ **snake_case**: Nomes em português
- ✅ **Docstrings**: Toda função documentada
- ✅ **Type Hints**: Anotações de tipo
- ✅ **Logging**: Sistema de logs completo
- ✅ **Modular**: Separação clara de responsabilidades
- ✅ **DRY**: Sem repetição de código

## 🤝 Contribuindo

Para adicionar funcionalidades:

1. **Criar serviço** em `tui/servicos/`
2. **Importar em** `tui/servicos/__init__.py`
3. **Usar na tela** via injeção de dependência
4. **Registrar tela** em `tui/app.py`
5. **Atualizar** este guia

## 📞 Contato/Suporte

Logs salvos em: `tui.log`

Para debug:
```bash
python tui_interactive.py 2>&1 | tee tui_debug.log
```

---

**Versão:** 1.0
**Última atualização:** 2026-04-14
**Status:** ✅ Funcional e pronto para uso
