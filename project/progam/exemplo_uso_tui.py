#!/usr/bin/env python
"""
Exemplos de uso da TUI e seus serviços.

Demonstra como usar a TUI e seus componentes programaticamente.
"""

import logging

logging.basicConfig(level=logging.INFO)


def exemplo_1_detecao_hardware():
    """Exemplo 1: Detectar hardware e obter recomendações."""
    print("\n" + "=" * 80)
    print("EXEMPLO 1: Detecção de Hardware")
    print("=" * 80)

    from tui.servicos import detectar_hardware

    perfil = detectar_hardware()

    print(f"\n🖥️  CPU:")
    print(f"   Nome: {perfil.cpu.nome}")
    print(f"   Cores Físicos: {perfil.cpu.nucleos_fisicos}")
    print(f"   Cores Lógicos: {perfil.cpu.nucleos_logicos}")
    print(f"   Frequência: {perfil.cpu.frequencia_mhz:.0f} MHz")

    print(f"\n🧠 Memória:")
    print(f"   Total: {perfil.memoria.total_gb:.2f} GB")
    print(f"   Disponível: {perfil.memoria.disponivel_gb:.2f} GB")
    print(f"   Uso: {perfil.memoria.percentual_uso:.1f}%")

    print(f"\n🎮 GPU:")
    if perfil.tem_gpu:
        for gpu in perfil.gpus:
            print(f"   [{gpu.indice}] {gpu.nome}")
            print(f"       Memória: {gpu.memoria_total_gb:.2f} GB")
    else:
        print("   Não disponível")

    print(f"\n💡 Recomendações:")
    print(f"   Dispositivo: {perfil.dispositivo_recomendado}")
    print(f"   Batch Size: {perfil.batch_size_recomendado}")
    print(f"   DataLoader Workers: {perfil.workers_recomendados}")


def exemplo_2_listar_datasets():
    """Exemplo 2: Listar datasets disponíveis."""
    print("\n" + "=" * 80)
    print("EXEMPLO 2: Listar Datasets")
    print("=" * 80)

    from tui.servicos import GerenciadorDatasets, TipoDataset

    gerenciador = GerenciadorDatasets()
    datasets = gerenciador.listar_datasets()

    print(f"\n📊 Total de datasets: {len(datasets)}\n")

    # Agrupar por tipo
    brutos = [d for d in datasets if d.tipo == TipoDataset.BRUTO]
    balanceados = [d for d in datasets if d.tipo == TipoDataset.BALANCEADO]

    if brutos:
        print("Datasets Brutos:")
        for ds in brutos:
            print(f"  ✓ {ds.nome}")
            print(f"    Tamanho: {ds.tamanho_mb:.1f} MB")
            print(f"    Descrição: {ds.descricao}")

    if balanceados:
        print("\nDatasets Balanceados:")
        for ds in balanceados:
            tecnica = ds.tecnica_balanceamento or "desconhecida"
            print(f"  ⚖️  {ds.nome} ({tecnica})")
            print(f"    Tamanho: {ds.tamanho_mb:.1f} MB")


def exemplo_3_listar_modelos():
    """Exemplo 3: Listar modelos disponíveis."""
    print("\n" + "=" * 80)
    print("EXEMPLO 3: Listar Modelos")
    print("=" * 80)

    from tui.servicos import GerenciadorModelos

    gerenciador = GerenciadorModelos()
    modelos = gerenciador.listar_modelos()

    print(f"\n🤖 Total de modelos: {len(modelos)}\n")

    # Agrupar por arquitetura
    agrupados = gerenciador.agrupar_por_arquitetura()

    for arquitetura, modelos_arquitetura in sorted(agrupados.items()):
        print(f"{arquitetura}:")
        for modelo in modelos_arquitetura:
            print(f"  • {modelo.nome}")
            print(f"    Classe: {modelo.classe_nome}")
            print(f"    Variante: {modelo.variante}")
            print(f"    XAI: {modelo.xai_metodo}")


def exemplo_4_resumo_modelo():
    """Exemplo 4: Obter resumo detalhado de um modelo."""
    print("\n" + "=" * 80)
    print("EXEMPLO 4: Resumo Detalhado de Modelo")
    print("=" * 80)

    from tui.servicos import GerenciadorModelos

    gerenciador = GerenciadorModelos()

    print("\nCNN Light:")
    print(gerenciador.obter_resumo("cnn_light"))

    print("\n" + "-" * 80)
    print("\nViT Robust:")
    print(gerenciador.obter_resumo("vit_robust"))


def exemplo_5_configuracao_experimento():
    """Exemplo 5: Criar configuração de experimento programaticamente."""
    print("\n" + "=" * 80)
    print("EXEMPLO 5: Configuração de Experimento")
    print("=" * 80)

    from tui import ConfiguracaoExperimento
    from tui.servicos import detectar_hardware

    # Criar configuração
    config = ConfiguracaoExperimento(
        dataset_selecionado="Galaxy10_SDSS",
        modelo_selecionado="cnn_light",
        num_epochs=20,
        batch_size=32,
        learning_rate=0.001,
    )

    # Obter recomendações
    perfil = detectar_hardware()

    print("\n📋 Configuração:")
    print(f"   Dataset: {config.dataset_selecionado}")
    print(f"   Modelo: {config.modelo_selecionado}")
    print(f"   Épocas: {config.num_epochs}")
    print(f"   Batch Size: {config.batch_size}")
    print(f"   Learning Rate: {config.learning_rate}")

    print("\n💡 Hardware Recomendado:")
    print(f"   Dispositivo: {perfil.dispositivo_recomendado}")
    print(f"   Workers: {perfil.workers_recomendados}")

    print("\n✓ Configuração completa?" f" {config.esta_completa()}")


def exemplo_6_executar_tui():
    """Exemplo 6: Executar a TUI interativa."""
    print("\n" + "=" * 80)
    print("EXEMPLO 6: Executar TUI Interativa")
    print("=" * 80)

    print("\nPara executar a TUI, use:")
    print("  python tui_interactive.py")
    print("\nOu programaticamente:")
    print("  from tui import executar")
    print("  executar()")


def exemplo_7_criar_app_customizada():
    """Exemplo 7: Criar aplicação com configuração customizada."""
    print("\n" + "=" * 80)
    print("EXEMPLO 7: App Customizada")
    print("=" * 80)

    from tui import criar_app, ConfiguracaoExperimento

    # Criar app
    app = criar_app()

    # Customizar configuração inicial
    app.config = ConfiguracaoExperimento(
        dataset_selecionado="Galaxy10_DECals",
        modelo_selecionado="vit_light",
        num_epochs=50,
        batch_size=64,
        learning_rate=0.0001,
    )

    print("\nApp criada com configuração customizada:")
    print(f"  Dataset: {app.config.dataset_selecionado}")
    print(f"  Modelo: {app.config.modelo_selecionado}")
    print(f"  Épocas: {app.config.num_epochs}")

    # Se descomentar a linha abaixo, a TUI vai iniciar
    # app.run()


if __name__ == "__main__":
    import sys

    print("🌌 Exemplos de Uso da TUI\n")

    exemplos = [
        ("1", "Detecção de Hardware", exemplo_1_detecao_hardware),
        ("2", "Listar Datasets", exemplo_2_listar_datasets),
        ("3", "Listar Modelos", exemplo_3_listar_modelos),
        ("4", "Resumo de Modelo", exemplo_4_resumo_modelo),
        ("5", "Configuração de Experimento", exemplo_5_configuracao_experimento),
        ("6", "Executar TUI", exemplo_6_executar_tui),
        ("7", "App Customizada", exemplo_7_criar_app_customizada),
    ]

    print("Exemplos disponíveis:\n")
    for num, descricao, _ in exemplos:
        print(f"  {num}. {descricao}")

    if len(sys.argv) > 1:
        escolha = sys.argv[1]
    else:
        escolha = input("\nEscolha um exemplo (1-7) ou 'todos': ")

    if escolha == "todos":
        for num, _, func in exemplos:
            try:
                func()
            except Exception as e:
                print(f"\n❌ Erro no exemplo {num}: {e}")
    else:
        try:
            num = int(escolha)
            if 1 <= num <= len(exemplos):
                exemplos[num - 1][2]()
            else:
                print("Escolha inválida")
        except (ValueError, IndexError):
            print("Escolha inválida")
