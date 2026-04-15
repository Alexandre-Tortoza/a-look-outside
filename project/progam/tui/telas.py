"""
Telas interativas da TUI.

Implementa fluxo completo de seleção de datasets, modelos e configuração.
"""

import logging
from typing import List, Optional

from textual.app import ComposeResult, Screen
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.widgets import (
    Header,
    Footer,
    Static,
    Button,
    Select,
    Input,
    Label,
    Checkbox,
    Switch,
    ProgressBar,
    RichLog,
)
from textual.screen import Screen as TextualScreen

from .servicos import (
    GerenciadorDatasets,
    GerenciadorModelos,
    detectar_hardware,
)

logger = logging.getLogger(__name__)


class TelaBoasVindas(TextualScreen):
    """Tela inicial com menu principal."""

    BINDINGS = [("q", "sair", "Sair")]

    def compose(self) -> ComposeResult:
        """Compor widgets da tela."""
        yield Header(show_clock=True)
        yield Container(
            Vertical(
                Static("Galaxy Morphology Classification", id="titulo"),
                Static("", id="espaco"),
                Static("Bem-vindo ao sistema de treinamento!", id="subtitulo"),
                Static("", id="espaco2"),
                Button("Novo Experimento", id="btn_novo", variant="primary"),
                Button("Listar Datasets", id="btn_datasets"),
                Button("Listar Modelos", id="btn_modelos"),
                Button("Hardware", id="btn_hardware"),
                Button("Sair", id="btn_sair", variant="error"),
                id="menu",
            ),
            id="content",
        )
        yield Footer()

    def on_button_pressed(self, evento) -> None:
        """Tratar clique em botões."""
        if evento.button.id == "btn_novo":
            self.app.push_screen("selecao_dataset")
        elif evento.button.id == "btn_datasets":
            self.app.push_screen("listar_datasets")
        elif evento.button.id == "btn_modelos":
            self.app.push_screen("listar_modelos")
        elif evento.button.id == "btn_hardware":
            self.app.push_screen("info_hardware")
        elif evento.button.id == "btn_sair":
            self.app.exit()

    def action_sair(self) -> None:
        """Ação de sair."""
        self.app.exit()


class TelaSelecaoDataset(TextualScreen):
    """Tela para seleção de dataset."""

    BINDINGS = [("escape", "voltar", "Voltar")]

    def __init__(self):
        super().__init__()
        self.gerenciador = GerenciadorDatasets()
        self.datasets = self.gerenciador.listar_datasets()

    def compose(self) -> ComposeResult:
        """Compor widgets da tela."""
        yield Header()
        yield Container(
            Vertical(
                Static("Selecione um Dataset", id="titulo"),
                Static(
                    f"Encontrados {len(self.datasets)} dataset(s)",
                    id="info_datasets",
                ),
                Static("", id="espaco"),
                self._criar_botoes_datasets(),
                Static("", id="espaco2"),
                Button("Gerar novo dataset", id="btn_gerar", variant="warning"),
                Button("Voltar", id="btn_voltar"),
                id="container",
            ),
            id="content",
        )
        yield Footer()

    def _criar_botoes_datasets(self):
        """Criar botões para cada dataset."""
        return ScrollableContainer(
            *[
                Button(
                    f"{'[BRUTO]' if ds.tipo.value == 'bruto' else '[BAL]'} {ds.nome} ({ds.tamanho_mb:.1f}MB)",
                    id=f"dataset_{ds.nome}",
                )
                for ds in self.datasets
            ]
        )

    def on_button_pressed(self, evento) -> None:
        """Tratar seleção de dataset."""
        if evento.button.id == "btn_voltar":
            self.app.pop_screen()
        elif evento.button.id == "btn_gerar":
            self.app.push_screen("gerar_dataset")
        elif evento.button.id.startswith("dataset_"):
            nome = evento.button.id.replace("dataset_", "")
            self.app.config.dataset_selecionado = nome
            self.app.push_screen("selecao_modelo")

    def action_voltar(self) -> None:
        """Ação de voltar."""
        self.app.pop_screen()


class TelaSelecaoModelo(TextualScreen):
    """Tela para seleção de modelo."""

    BINDINGS = [("escape", "voltar", "Voltar")]

    def __init__(self):
        super().__init__()
        self.gerenciador = GerenciadorModelos()
        self.modelos = self.gerenciador.listar_modelos()

    def compose(self) -> ComposeResult:
        """Compor widgets da tela."""
        yield Header()
        yield Container(
            Vertical(
                Static("Selecione um Modelo", id="titulo"),
                Static(
                    f"Encontrados {len(self.modelos)} modelo(s)",
                    id="info_modelos",
                ),
                Static("", id="espaco"),
                self._criar_botoes_modelos(),
                Static("", id="espaco2"),
                Button("Voltar", id="btn_voltar"),
                id="container",
            ),
            id="content",
        )
        yield Footer()

    def _criar_botoes_modelos(self):
        """Criar botões para cada modelo."""
        return ScrollableContainer(
            *[
                Button(
                    f"{modelo.classe_nome} ({modelo.variante})",
                    id=f"modelo_{modelo.nome}",
                )
                for modelo in self.modelos
            ]
        )

    def on_button_pressed(self, evento) -> None:
        """Tratar seleção de modelo."""
        if evento.button.id == "btn_voltar":
            self.app.pop_screen()
        elif evento.button.id.startswith("modelo_"):
            nome = evento.button.id.replace("modelo_", "")
            self.app.config.modelo_selecionado = nome
            self.app.push_screen("configuracao_parametros")

    def action_voltar(self) -> None:
        """Ação de voltar."""
        self.app.pop_screen()


class TelaConfiguracaoParametros(TextualScreen):
    """Tela para configuração de parâmetros de treinamento."""

    BINDINGS = [("escape", "voltar", "Voltar")]

    def compose(self) -> ComposeResult:
        """Compor widgets da tela."""
        perfil_hw = detectar_hardware()
        eh_vit = self.app.config.modelo_selecionado and "vit" in self.app.config.modelo_selecionado.lower()

        yield Header()
        yield Container(
            ScrollableContainer(
                Vertical(
                    Static("Configuração de Parâmetros", id="titulo"),
                    Static("", id="espaco"),
                    Static("Parâmetros de Treinamento:", id="header_params"),
                    Input(
                        placeholder="10",
                        id="input_epochs",
                        value="10",
                    ),
                    Static("Épocas", id="label_epochs"),
                    Input(
                        placeholder="32",
                        id="input_batch_size",
                        value="32",
                    ),
                    Static("Batch Size", id="label_batch_size"),
                    Input(
                        placeholder="0.001",
                        id="input_lr",
                        value="0.001",
                    ),
                    Static("Learning Rate", id="label_lr"),
                    Input(
                        placeholder="0.7",
                        id="input_split",
                        value="0.7",
                    ),
                    Static("Train/Val Split (0.0-1.0)", id="label_split"),
                    Static("", id="espaco_opts"),
                    Static("Opções Avançadas:", id="header_advanced"),
                    *(
                        [
                            Select(
                                [("Scratch (treinar do zero)", "scratch"), ("Fine-tuning (pré-treinado)", "finetune")],
                                value="scratch",
                                id="select_vit_mode",
                            ),
                            Static("Modo ViT", id="label_vit_mode"),
                        ]
                        if eh_vit
                        else []
                    ),
                    Static("Early Stopping", id="label_early_stop"),
                    Horizontal(
                        Static("Ativo:", id="label_early_active"),
                        Switch(id="switch_early_stop", value=True),
                        id="container_early_switch",
                    ),
                    Input(
                        placeholder="5",
                        id="input_early_paciencia",
                        value="5",
                    ),
                    Static("Paciência (épocas)", id="label_early_paciencia"),
                    Static("", id="espaco_ckpt"),
                    Horizontal(
                        Static("Salvar Checkpoints:", id="label_ckpt"),
                        Switch(id="switch_checkpoints", value=True),
                        id="container_ckpt_switch",
                    ),
                    Static("", id="espaco2"),
                    Static("Configuração de Hardware:", id="header_hw"),
                    Static(
                        f"CPU: {perfil_hw.cpu.nome} ({perfil_hw.cpu.nucleos_fisicos} cores)",
                        id="info_cpu",
                    ),
                    Static(
                        f"Memória: {perfil_hw.memoria.total_gb:.1f}GB "
                        f"({perfil_hw.memoria.disponivel_gb:.1f}GB disponível)",
                        id="info_memory",
                    ),
                    Static(
                        f"GPU: {'Disponível' if perfil_hw.tem_gpu else 'Não disponível'}",
                        id="info_gpu",
                    ),
                    Static("", id="espaco3"),
                    Static("Recomendações automáticas:", id="header_recom"),
                    Static(
                        f"Dispositivo: {perfil_hw.dispositivo_recomendado}",
                        id="recom_device",
                    ),
                    Static(
                        f"Batch Size: {perfil_hw.batch_size_recomendado}",
                        id="recom_batch",
                    ),
                    Static(
                        f"Workers: {perfil_hw.workers_recomendados}",
                        id="recom_workers",
                    ),
                    id="form_container",
                )
            ),
            id="content",
        )
        with Horizontal(id="buttons"):
            yield Button("Continuar", id="btn_continuar", variant="primary")
            yield Button("Voltar", id="btn_voltar")
        yield Footer()

    def on_button_pressed(self, evento) -> None:
        """Tratar clique em botões."""
        if evento.button.id == "btn_continuar":
            self._salvar_parametros()
            self.app.push_screen("confirmacao_experimento")
        elif evento.button.id == "btn_voltar":
            self.app.pop_screen()

    def _salvar_parametros(self):
        """Salvar parâmetros configurados."""
        try:
            input_epochs = self.query_one("#input_epochs", Input)
            input_batch = self.query_one("#input_batch_size", Input)
            input_lr = self.query_one("#input_lr", Input)
            input_split = self.query_one("#input_split", Input)
            switch_early = self.query_one("#switch_early_stop", Switch)
            input_paciencia = self.query_one("#input_early_paciencia", Input)
            switch_ckpt = self.query_one("#switch_checkpoints", Switch)

            self.app.config.num_epochs = int(input_epochs.value or "10")
            self.app.config.batch_size = int(input_batch.value or "32")
            self.app.config.learning_rate = float(input_lr.value or "0.001")
            self.app.config.divisao_treino = float(input_split.value or "0.7")
            self.app.config.early_stop_ativo = switch_early.value
            self.app.config.early_stop_paciencia = int(input_paciencia.value or "5")
            self.app.config.salvar_checkpoints = switch_ckpt.value

            # Se for ViT, salva modo
            eh_vit = self.app.config.modelo_selecionado and "vit" in self.app.config.modelo_selecionado.lower()
            if eh_vit:
                try:
                    select_modo = self.query_one("#select_vit_mode", Select)
                    self.app.config.modo_vit = select_modo.value or "scratch"
                except:
                    self.app.config.modo_vit = "scratch"

        except Exception as e:
            logger.error(f"Erro ao salvar parâmetros: {e}")

    def action_voltar(self) -> None:
        """Ação de voltar."""
        self.app.pop_screen()


class TelaConfirmacaoExperimento(TextualScreen):
    """Tela de confirmação antes de executar."""

    BINDINGS = [("escape", "voltar", "Voltar")]

    def compose(self) -> ComposeResult:
        """Compor widgets da tela."""
        yield Header()
        yield Container(
            Vertical(
                Static("Confirmação do Experimento", id="titulo"),
                Static("", id="espaco"),
                Static(self._gerar_resumo(), id="resumo"),
                Static("", id="espaco2"),
                Static("Clique em 'Executar' para começar o treinamento", id="aviso"),
                Button("Executar Experimento", id="btn_executar", variant="success"),
                Button("Voltar", id="btn_voltar"),
                id="container",
            ),
            id="content",
        )
        yield Footer()

    def _gerar_resumo(self) -> str:
        """Gerar resumo da configuração."""
        config = self.app.config
        linhas = [
            "Resumo do Experimento:",
            "",
            f"Dataset: {config.dataset_selecionado}",
            f"Modelo: {config.modelo_selecionado}",
            f"Épocas: {config.num_epochs}",
            f"Batch Size: {config.batch_size}",
            f"Learning Rate: {config.learning_rate}",
        ]
        return "\n".join(linhas)

    def on_button_pressed(self, evento) -> None:
        """Tratar botões."""
        if evento.button.id == "btn_executar":
            self.app.push_screen("executando_experimento")
        elif evento.button.id == "btn_voltar":
            self.app.pop_screen()

    def action_voltar(self) -> None:
        """Ação de voltar."""
        self.app.pop_screen()


class TelaListarDatasets(TextualScreen):
    """Tela para listar datasets sem selecionar."""

    BINDINGS = [("escape", "voltar", "Voltar")]

    def __init__(self):
        super().__init__()
        self.gerenciador = GerenciadorDatasets()
        self.datasets = self.gerenciador.listar_datasets()

    def compose(self) -> ComposeResult:
        """Compor widgets da tela."""
        yield Header()
        yield Container(
            ScrollableContainer(
                Vertical(
                    Static("Datasets Disponíveis", id="titulo"),
                    Static(f"Total: {len(self.datasets)}", id="info"),
                    Static("", id="espaco"),
                    *[
                        Static(
                            f"{ds.nome}\n{ds.descricao}\nTamanho: {ds.tamanho_mb:.1f}MB",
                            id=f"info_dataset_{ds.nome}",
                        )
                        for ds in self.datasets
                    ],
                    id="lista_datasets",
                )
            ),
            id="content",
        )
        with Horizontal(id="buttons"):
            yield Button("Voltar", id="btn_voltar")
        yield Footer()

    def on_button_pressed(self, evento) -> None:
        """Tratar botão."""
        if evento.button.id == "btn_voltar":
            self.app.pop_screen()

    def action_voltar(self) -> None:
        """Ação de voltar."""
        self.app.pop_screen()


class TelaListarModelos(TextualScreen):
    """Tela para listar modelos sem selecionar."""

    BINDINGS = [("escape", "voltar", "Voltar")]

    def __init__(self):
        super().__init__()
        self.gerenciador = GerenciadorModelos()
        self.modelos = self.gerenciador.listar_modelos()

    def compose(self) -> ComposeResult:
        """Compor widgets da tela."""
        yield Header()
        yield Container(
            ScrollableContainer(
                Vertical(
                    Static("Modelos Disponíveis", id="titulo"),
                    Static(f"Total: {len(self.modelos)}", id="info"),
                    Static("", id="espaco"),
                    *[
                        Static(
                            f"{m.classe_nome} ({m.variante})\nXAI: {m.xai_metodo}",
                            id=f"info_modelo_{m.nome}",
                        )
                        for m in self.modelos
                    ],
                    id="lista_modelos",
                )
            ),
            id="content",
        )
        with Horizontal(id="buttons"):
            yield Button("Voltar", id="btn_voltar")
        yield Footer()

    def on_button_pressed(self, evento) -> None:
        """Tratar botão."""
        if evento.button.id == "btn_voltar":
            self.app.pop_screen()

    def action_voltar(self) -> None:
        """Ação de voltar."""
        self.app.pop_screen()


class TelaInfoHardware(TextualScreen):
    """Tela com informações detalhadas de hardware."""

    BINDINGS = [("escape", "voltar", "Voltar")]

    def compose(self) -> ComposeResult:
        """Compor widgets da tela."""
        perfil = detectar_hardware()

        yield Header()
        yield Container(
            ScrollableContainer(
                Vertical(
                    Static("Informações de Hardware", id="titulo"),
                    Static("", id="espaco"),
                    Static("CPU:", id="header_cpu"),
                    Static(f"  Nome: {perfil.cpu.nome}", id="info_cpu_nome"),
                    Static(f"  Cores físicos: {perfil.cpu.nucleos_fisicos}", id="info_cpu_cores"),
                    Static(f"  Cores lógicos: {perfil.cpu.nucleos_logicos}", id="info_cpu_logical"),
                    Static(f"  Frequência: {perfil.cpu.frequencia_mhz:.0f} MHz", id="info_cpu_freq"),
                    Static("", id="espaco2"),
                    Static("Memória:", id="header_mem"),
                    Static(f"  Total: {perfil.memoria.total_gb:.2f} GB", id="info_mem_total"),
                    Static(
                        f"  Disponível: {perfil.memoria.disponivel_gb:.2f} GB",
                        id="info_mem_avail",
                    ),
                    Static(
                        f"  Uso: {perfil.memoria.percentual_uso:.1f}%",
                        id="info_mem_uso",
                    ),
                    Static("", id="espaco3"),
                    Static(
                        f"GPU: {'Disponível' if perfil.tem_gpu else 'Não disponível'}",
                        id="header_gpu",
                    ),
                    *(
                        [
                            Static(f"  {gpu.nome}", id=f"info_gpu_nome_{gpu.indice}"),
                            Static(
                                f"  Memória: {gpu.memoria_total_gb:.2f} GB",
                                id=f"info_gpu_mem_{gpu.indice}",
                            ),
                        ]
                        for gpu in perfil.gpus
                    ),
                    Static("", id="espaco4"),
                    Static("Recomendações:", id="header_recom"),
                    Static(
                        f"  Dispositivo: {perfil.dispositivo_recomendado}",
                        id="recom_device",
                    ),
                    Static(
                        f"  Batch Size: {perfil.batch_size_recomendado}",
                        id="recom_batch",
                    ),
                    Static(
                        f"  DataLoader Workers: {perfil.workers_recomendados}",
                        id="recom_workers",
                    ),
                    id="info_container",
                )
            ),
            id="content",
        )
        with Horizontal(id="buttons"):
            yield Button("Voltar", id="btn_voltar")
        yield Footer()

    def on_button_pressed(self, evento) -> None:
        """Tratar botão."""
        if evento.button.id == "btn_voltar":
            self.app.pop_screen()

    def action_voltar(self) -> None:
        """Ação de voltar."""
        self.app.pop_screen()


class TelaGerarDataset(TextualScreen):
    """Tela placeholder para gerar novos datasets."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Vertical(
                Static("Gerar Novo Dataset", id="titulo"),
                Static("Funcionalidade em desenvolvimento...", id="status"),
                Button("Voltar", id="btn_voltar"),
                id="container",
            ),
            id="content",
        )
        yield Footer()

    def on_button_pressed(self, evento) -> None:
        if evento.button.id == "btn_voltar":
            self.app.pop_screen()


class TelaExecutandoExperimento(TextualScreen):
    """Tela de execução do treinamento com progresso em tempo real."""

    BINDINGS = [("escape", "cancelar", "Cancelar")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Vertical(
                Static("Treinamento em Execução", id="titulo"),
                Static("", id="espaco"),
                ProgressBar(id="progress_bar", total=100),
                Static("", id="espaco2"),
                Static("Logs de Treinamento:", id="header_logs"),
                RichLog(id="log_output", markup=True),
                Static("", id="espaco3"),
                Button("Voltar", id="btn_voltar", variant="primary"),
                id="container",
            ),
            id="content",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Iniciar treinamento quando tela é montada."""
        self.run_worker(self._executar_treino, exclusive=True)

    def _executar_treino(self) -> None:
        """Worker para executar treinamento em background."""
        from pathlib import Path
        from models import get_model
        from datasets.preparador import PreparadorDados
        from models.treinador import TreinadorModelo

        log_widget = self.query_one("#log_output", RichLog)
        progress_widget = self.query_one("#progress_bar", ProgressBar)

        def fn_log(msg: str):
            """Callback para log."""
            self.call_from_thread(lambda: log_widget.write(msg))

        def fn_progresso(epoca: int, total: int):
            """Callback para progresso."""
            self.call_from_thread(lambda: progress_widget.update(progress=epoca / total * 100))

        try:
            config = self.app.config
            fn_log(f"[bold]Iniciando treinamento[/bold]")
            fn_log(f"Dataset: {config.dataset_selecionado}")
            fn_log(f"Modelo: {config.modelo_selecionado}")
            fn_log(f"Épocas: {config.num_epochs}")
            fn_log(f"Learning Rate: {config.learning_rate}")
            fn_log(f"Early Stop: {config.early_stop_ativo} (paciência: {config.early_stop_paciencia})")
            fn_log("")

            # Criar diretório de saída (run-XX)
            results_dir = Path(__file__).parent.parent / "results"
            runs = [d for d in results_dir.glob("run-*") if d.is_dir()]
            run_num = max([int(d.name.split("-")[1]) for d in runs]) + 1 if runs else 0
            dir_saida = results_dir / f"run-{run_num:02d}"
            dir_saida.mkdir(parents=True, exist_ok=True)
            fn_log(f"Diretório de saída: {dir_saida}")

            # Carregar dados
            fn_log("[yellow]Carregando dataset...[/yellow]")
            dataset_path = Path(__file__).parent.parent / "datasets" / f"{config.dataset_selecionado}"
            preparador = PreparadorDados()
            loader_treino, loader_val = preparador.preparar(
                caminho_dataset=str(dataset_path),
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                divisao_treino=config.divisao_treino,
                img_size=64,  # Redimensiona para 64x64 (compatível com ViT)
            )
            fn_log(f"✓ Dataset carregado ({len(loader_treino)} batches treino, {len(loader_val)} batches val)")

            # Instanciar modelo
            fn_log("[yellow]Instanciando modelo...[/yellow]")
            wrapper = get_model(config.modelo_selecionado)

            # Se for ViT, passar modo_treino
            if hasattr(wrapper, "modo_treino"):
                wrapper.modo_treino = config.modo_vit

            rede = wrapper.build(num_classes=10, img_size=64)
            fn_log(f"✓ Modelo {wrapper.name} ({wrapper.variant}) instanciado")

            # Treinar
            fn_log("[bold green]Iniciando treinamento...[/bold green]")
            fn_log("")

            treinador = TreinadorModelo(
                num_epochs=config.num_epochs,
                learning_rate=config.learning_rate,
                dispositivo=config.dispositivo,
                dir_saida=dir_saida,
                fn_log=fn_log,
                fn_progresso=fn_progresso,
                early_stop_ativo=config.early_stop_ativo,
                early_stop_paciencia=config.early_stop_paciencia,
                salvar_checkpoints=config.salvar_checkpoints,
            )

            metricas = treinador.treinar(
                rede,
                nome_modelo=config.modelo_selecionado,
                loader_treino=loader_treino,
                loader_val=loader_val,
            )

            fn_log("")
            fn_log("[bold green]✓ Treinamento concluído![/bold green]")
            fn_log(f"Épocas executadas: {metricas['epocas_executadas']}")
            fn_log(f"Acurácia final: {metricas['accuracy']:.2%}")
            fn_log(f"Acurácia validação: {metricas['val_accuracy']:.2%}")
            fn_log(f"Loss final: {metricas['loss']:.4f}")
            fn_log(f"Modelo salvo em: {dir_saida / f'{config.modelo_selecionado}.pth'}")

        except Exception as e:
            fn_log(f"[bold red]✗ Erro durante treinamento:[/bold red]")
            fn_log(f"[red]{str(e)}[/red]")
            logger.exception(f"Erro no treinamento: {e}")

    def on_button_pressed(self, evento) -> None:
        """Tratar botão."""
        if evento.button.id == "btn_voltar":
            self.app.pop_screen()

    def action_cancelar(self) -> None:
        """Ação para cancelar (atalho ESC)."""
        self.app.pop_screen()
