"""
Gerenciador de modelos disponíveis.

Descobre, lista e fornece informações sobre modelos.
"""

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Dict, List

from models import get_model, list_models

logger = logging.getLogger(__name__)


@dataclass
class InfoModelo:
    """Informações sobre um modelo."""

    nome: str
    classe_nome: str
    variante: str
    xai_metodo: str
    descricao: str
    parametros_build: Dict[str, str] = None

    def __post_init__(self):
        if self.parametros_build is None:
            self.parametros_build = {}


class GerenciadorModelos:
    """Gerencia modelos de classificação disponíveis."""

    # Descrições dos modelos
    DESCRICOES = {
        "CNN": "Rede convolucional clássica. Light: rápida e leve. Robust: mais precisa.",
        "ViT": "Vision Transformer. Light: transformador leve. Robust: transformador completo.",
        "MobileNet": "Arquitetura otimizada para mobile. Light: leve. Robust: mais poderosa.",
        "EasyNet": "Arquitetura simplificada e didática. Light: baseline simples. Robust: versão estendida.",
    }

    def __init__(self):
        """Inicializar gerenciador."""
        self.logger = logger
        self._cache_modelos = None

    def listar_modelos(self) -> List[InfoModelo]:
        """
        Listar todos os modelos disponíveis.

        Returns:
            Lista com informações de cada modelo.
        """
        if self._cache_modelos is not None:
            return self._cache_modelos

        modelos = []

        for nome_modelo in list_models():
            try:
                info = self._extrair_info_modelo(nome_modelo)
                modelos.append(info)
                self.logger.debug(f"Modelo descoberto: {nome_modelo}")

            except Exception as e:
                self.logger.warning(f"Erro ao carregar modelo {nome_modelo}: {e}")

        self._cache_modelos = sorted(modelos, key=lambda m: m.nome)
        return self._cache_modelos

    def obter_modelo(self, nome: str) -> InfoModelo:
        """
        Obter informações específicas de um modelo.

        Args:
            nome: Nome do modelo.

        Returns:
            Informações do modelo.

        Raises:
            ValueError: Se modelo não encontrado.
        """
        for modelo in self.listar_modelos():
            if modelo.nome == nome:
                return modelo

        raise ValueError(f"Modelo {nome} não encontrado")

    def _extrair_info_modelo(self, nome_modelo: str) -> InfoModelo:
        """
        Extrair informações de um modelo específico.

        Args:
            nome_modelo: Nome do modelo no registry.

        Returns:
            InfoModelo com dados extraídos.
        """
        instancia = get_model(nome_modelo)

        # Extrair nome da arquitetura (ex: "cnn_light" → "CNN")
        nome_arquitetura = instancia.name

        # Montar descrição
        descricao = self.DESCRICOES.get(
            nome_arquitetura,
            f"Modelo {nome_arquitetura}",
        )

        # Extrair parâmetros do método build
        parametros_build = self._extrair_parametros_build(instancia)

        return InfoModelo(
            nome=nome_modelo,
            classe_nome=instancia.__class__.__name__,
            variante=instancia.variant,
            xai_metodo=instancia.xai_method,
            descricao=descricao,
            parametros_build=parametros_build,
        )

    def _extrair_parametros_build(self, instancia: Any) -> Dict[str, str]:
        """
        Extrair parâmetros do método build da instância.

        Args:
            instancia: Instância do modelo.

        Returns:
            Dicionário {nome_param: tipo_esperado}.
        """
        try:
            assinatura = inspect.signature(instancia.build)
            parametros = {}

            for nome_param, param in assinatura.parameters.items():
                if nome_param == "self":
                    continue

                tipo_anotacao = (
                    param.annotation.__name__
                    if param.annotation != inspect.Parameter.empty
                    else "qualquer"
                )

                parametros[nome_param] = tipo_anotacao

            return parametros

        except Exception as e:
            self.logger.debug(f"Erro ao extrair parâmetros: {e}")
            return {}

    def agrupar_por_arquitetura(self) -> Dict[str, List[InfoModelo]]:
        """
        Agrupar modelos por arquitetura.

        Returns:
            Dicionário {arquitetura: [modelos]}.
        """
        agrupados: Dict[str, List[InfoModelo]] = {}

        for modelo in self.listar_modelos():
            chave = modelo.nome.split("_")[0].upper()

            if chave not in agrupados:
                agrupados[chave] = []

            agrupados[chave].append(modelo)

        return agrupados

    def obter_resumo(self, nome_modelo: str) -> str:
        """
        Obter resumo formatado de um modelo.

        Args:
            nome_modelo: Nome do modelo.

        Returns:
            String com resumo formatado.
        """
        try:
            info = self.obter_modelo(nome_modelo)

            linhas = [
                f"🤖 {info.classe_nome}",
                f"📊 Variante: {info.variante}",
                f"🎨 XAI: {info.xai_metodo}",
                f"📝 {info.descricao}",
            ]

            if info.parametros_build:
                linhas.append("\nParâmetros do treino:")
                for param, tipo in info.parametros_build.items():
                    linhas.append(f"  • {param}: {tipo}")

            return "\n".join(linhas)

        except Exception as e:
            return f"Erro ao obter resumo: {e}"
