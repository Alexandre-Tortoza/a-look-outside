# 🌌 Implementação Completa: Pipeline de Balanceamento + TUI

## 📋 Resumo Executivo

Foram implementados dois sistemas complementares para o projeto de classificação de morfologia galáctica:

1. **Pipeline de Balanceamento de Datasets** - Sistema modular para aplicar técnicas de balanceamento
2. **TUI Interativa** - Interface de usuário em texto completa para configuração de experimentos

---

## 📦 PARTE 1: Pipeline de Balanceamento

### Localização
```
progam/
├── main.py (modificado)
├── datasets/
│   ├── __init__.py
│   ├── carregador.py
│   ├── pipelines.py
│   ├── README.md
│   └── balanceadores/
│       ├── __init__.py
│       ├── base.py
│       ├── adasyn.py
│       ├── smote.py
│       ├── undersampling.py
│       ├── oversampling.py
│       ├── estratificacao.py
│       └── hibrido.py
```

### Funcionalidades

✅ **Carregamento de Datasets**
- Carrega H5 (SDSS e DECaLS)
- Tratamento de erros robusto
- Logging detalhado

✅ **6 Técnicas de Balanceamento**
- ADASYN: Geração sintética adaptativa
- SMOTE: Synthetic Minority Over-sampling
- Undersampling: Redução da maioria
- Oversampling: Duplicação aleatória
- Estratificação: Mantém proporções
- Híbrido: Combina múltiplas técnicas

✅ **Pipeline Completa**
- Orquestração de processos
- Salvamento automático em NPZ
- Logging de distribuições antes/depois

### Como Usar

```bash
# Todos os datasets, todas as técnicas
python main.py balanceamento --dataset ambos

# Apenas SDSS com técnicas específicas
python main.py balanceamento --dataset sdss --tecnicas smote undersampling

# DECaLS com ADASYN
python main.py balanceamento --dataset decals --tecnicas adasyn
```

### Exemplo de Distribuição Antes/Depois

```
Distribuição ANTES:
  {0: 3461, 1: 6997, 2: 6292, 3: 349, 4: 1534, 5: 17, ...}
  
Distribuição DEPOIS (SMOTE):
  {0: 6997, 1: 6997, 2: 6997, 3: 6997, 4: 6997, ...}
```

---

## 🌌 PARTE 2: TUI Interativa

### Localização
```
progam/
├── tui_interactive.py           # Script de entrada
├── GUIA_TUI.md
├── exemplo_uso_tui.py
└── tui/
    ├── __init__.py
    ├── app.py
    ├── telas.py
    └── servicos/
        ├── __init__.py
        ├── detecao_hardware.py
        ├── gerenciador_datasets.py
        └── gerenciador_modelos.py
```

### 8 Telas Implementadas

1. **Tela de Boas-vindas** - Menu principal
2. **Seleção de Dataset** - Escolher ou gerar dataset
3. **Seleção de Modelo** - Escolher arquitetura
4. **Configuração de Parâmetros** - Ajustar hiperparâmetros
5. **Confirmação** - Revisar antes de executar
6. **Listar Datasets** - Visão geral
7. **Listar Modelos** - Visão geral
8. **Informações de Hardware** - Especificações do sistema

### 3 Serviços Modulares

#### 1. Detecção de Hardware (`detecao_hardware.py`)

```python
from tui.servicos import detectar_hardware

perfil = detectar_hardware()
print(f"CPU: {perfil.cpu.nome}")
print(f"Cores: {perfil.cpu.nucleos_fisicos}")
print(f"Memória: {perfil.memoria.total_gb}GB")
print(f"GPU: {'Sim' if perfil.tem_gpu else 'Não'}")

# Recomendações
print(f"Dispositivo: {perfil.dispositivo_recomendado}")
print(f"Batch Size: {perfil.batch_size_recomendado}")
print(f"Workers: {perfil.workers_recomendados}")
```

#### 2. Gerenciador de Datasets (`gerenciador_datasets.py`)

```python
from tui.servicos import GerenciadorDatasets

gerenciador = GerenciadorDatasets()
datasets = gerenciador.listar_datasets()

for ds in datasets:
    print(f"{ds.nome}: {ds.tamanho_mb}MB ({ds.tipo.value})")

# Gerar novo dataset balanceado
gerenciador.gerar_balanceado(
    "Galaxy10_SDSS",
    ["smote", "undersampling"]
)
```

#### 3. Gerenciador de Modelos (`gerenciador_modelos.py`)

```python
from tui.servicos import GerenciadorModelos

gerenciador = GerenciadorModelos()
modelos = gerenciador.listar_modelos()

for modelo in modelos:
    print(f"{modelo.nome}: {modelo.classe_nome} ({modelo.variante})")

# Agrupar por arquitetura
agrupados = gerenciador.agrupar_por_arquitetura()

# Resumo formatado
print(gerenciador.obter_resumo("cnn_light"))
```

### Como Executar

```bash
# Opção 1: Script direto
python tui_interactive.py

# Opção 2: Programaticamente
python -c "from tui import executar; executar()"

# Opção 3: Exemplos
python exemplo_uso_tui.py
```

### Fluxo de Uso Típico

```
1. Inicia com python tui_interactive.py
2. Vê menu principal
3. Escolhe "Novo Experimento"
4. Seleciona dataset (ex: Galaxy10_SDSS)
5. Seleciona modelo (ex: cnn_light)
6. Configura parâmetros (epochs, batch_size, lr)
7. Hardware detectado e exibe recomendações
8. Confirma e executa
```

---

## 🏗️ Arquitetura Geral

### Diagrama de Componentes

```
┌─────────────────────────────────────────────┐
│         TUI Interativa (Textual)            │
├─────────────────────────────────────────────┤
│  app.py (AplicacaoGalaxy)                   │
│  telas.py (8 telas)                         │
└─────────┬───────────────────────────────────┘
          │
    ┌─────┴──────────┬──────────────┬──────────┐
    │                │              │          │
    ▼                ▼              ▼          ▼
┌─────────────┐ ┌──────────┐ ┌─────────┐ ┌──────────────┐
│ Detecção    │ │ Gerenc.  │ │ Gerenc. │ │ Configuração │
│ Hardware    │ │ Datasets │ │ Modelos │ │ Experimento  │
└─────┬───────┘ └────┬─────┘ └────┬────┘ └──────┬───────┘
      │              │            │             │
      └──────────────┼────────────┼─────────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
      ┌─────────────┐     ┌──────────────┐
      │ Pipeline de │     │ Registry de  │
      │ Balanceamento      │ Modelos      │
      └─────────────┘     └──────────────┘
```

### Dependências Entre Módulos

```
datasets/
  ├── carregador.py
  ├── pipelines.py
  │   └── importa: balanceadores/
  └── balanceadores/
      ├── base.py (classe abstrata)
      ├── adasyn.py, smote.py, ... (concretos)
      └── hibrido.py (orquestra)

models/
  ├── __init__.py (registry)
  ├── base.py (GalaxyClassifier)
  └── *.py (implementações)

tui/
  ├── app.py (Textual App)
  ├── telas.py (Screen subclasses)
  └── servicos/
      ├── detecao_hardware.py
      ├── gerenciador_datasets.py
      └── gerenciador_modelos.py
          └── importa: models/
```

---

## 📊 Estrutura de Dados

### ConfiguracaoExperimento

```python
@dataclass
class ConfiguracaoExperimento:
    dataset_selecionado: Optional[str] = None
    modelo_selecionado: Optional[str] = None
    tecnicas_balanceamento: list = field(default_factory=list)
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    dispositivo: str = "auto"
    num_workers: int = 4
```

### PerfilHardware

```python
@dataclass
class PerfilHardware:
    cpu: InfoCPU
    memoria: InfoMemoria
    gpus: List[InfoGPU] = field(default_factory=list)
    
    # Propriedades computadas
    @property
    def dispositivo_recomendado(self) -> str: ...
    
    @property
    def batch_size_recomendado(self) -> int: ...
    
    @property
    def workers_recomendados(self) -> int: ...
```

### InfoDataset

```python
@dataclass
class InfoDataset:
    nome: str
    tipo: TipoDataset  # BRUTO ou BALANCEADO
    caminho: Path
    tamanho_mb: float
    descricao: str
    num_amostras: Optional[int] = None
    num_classes: Optional[int] = None
    tecnica_balanceamento: Optional[str] = None
```

---

## 🎯 Requisitos Atendidos

### ✅ Pipeline de Balanceamento

- [x] Fluxo geral com análise
- [x] Opções para configurar datasets-produtos
- [x] Técnicas diferentes de balanceamento
- [x] Geração de múltiplos datasets (ADASYN, SMOTE, etc)
- [x] Tudo em português (pt-br)
- [x] Snake_case
- [x] Arquivos modularizados e separados
- [x] Sistema desacoplado

### ✅ TUI Interativa

- [x] Seleção dinâmica de datasets
- [x] Geração automática de datasets se não existirem
- [x] Seleção dinâmica de modelos
- [x] Configuração de parâmetros dinâmica
- [x] Detecção automática de hardware
- [x] Sugestão de configurações
- [x] Override manual das configurações
- [x] Arquitetura modular e desacoplada
- [x] Fácil extensão para novos componentes
- [x] Código limpo em pt-BR com snake_case
- [x] Preparado para internacionalização

---

## 🚀 Como Começar

### 1. Instalar Dependências

```bash
pip install textual psutil scikit-learn numpy h5py torch
```

### 2. Testar Pipeline de Balanceamento

```bash
# Gerar dataset balanceado
python main.py balanceamento --dataset sdss --tecnicas smote undersampling

# Ver datasets gerados
ls datasets/balanceados/
```

### 3. Executar TUI Interativa

```bash
# Modo direto
python tui_interactive.py

# Ver exemplos
python exemplo_uso_tui.py
```

### 4. Verificar Documentação

```bash
# Pipeline
cat datasets/README.md

# TUI
cat progam/GUIA_TUI.md
cat progam/TUI_README.md
```

---

## 📈 Próximas Implementações

### Pipeline de Balanceamento
- [ ] Suporte a mais técnicas (BorderlineSMOTE, SVMSMOTE)
- [ ] Análise estatística de balanceamento
- [ ] Comparação de técnicas

### TUI
- [ ] Execução com monitoramento em tempo real
- [ ] Dashboard de resultados
- [ ] Histórico de experimentos
- [ ] Exportação/Importação de configurações
- [ ] Integração com main.py via subcomando `tui`

---

## 📝 Notas Importantes

### Padrões de Código

- **Linguagem**: Português (pt-br) para docstrings e variáveis
- **Nomenclatura**: snake_case para funções e variáveis
- **Documentação**: Docstrings em todas as funções
- **Type Hints**: Anotações de tipo em todos os parâmetros
- **Logging**: Sistema de logging configurado em cada módulo

### Testado Com

- Python 3.14
- Textual 0.50+
- psutil 5.9+
- scikit-learn 1.4+
- h5py 3.16+
- numpy 1.24+
- torch 2.0+ (para GPU)

### Conhecidos Funcionando

- ✅ Dataset SDSS carrega (21.785 imagens)
- ✅ Dataset DECaLS carrega (8.671 imagens)
- ✅ Balanceadores implementados e testados
- ✅ Hardware auto-detectado
- ✅ 8 modelos descobertos automaticamente
- ✅ TUI inicia sem erros

---

## 🔗 Referências

- **Textual**: https://textual.textualize.io/
- **SMOTE**: https://imbalanced-learn.org/
- **PyTorch**: https://pytorch.org/
- **psutil**: https://psutil.readthedocs.io/

---

## 📞 Suporte

Para debug ou problemas:

1. Verificar `tui.log` (se houver)
2. Executar exemplos em `exemplo_uso_tui.py`
3. Ler documentação em `GUIA_TUI.md`

---

**Versão**: 1.0  
**Data**: 2026-04-14  
**Status**: ✅ Completo e funcional
