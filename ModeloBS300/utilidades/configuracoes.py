"""
Módulo de Configuração Global e Constantes do Sistema Financeiro
================================================================

Este módulo centraliza todas as configurações, hiperparâmetros e constantes
físicas/matemáticas utilizadas em todo o Super Modelo Black-Scholes.

Autor: Luiz Tiago Wilcke
Versão: 1.0.0
Data: 2026-01-19
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import json
import logging

# Configuração de Logging para Rastreabilidade Completa
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='sistema_financeiro.log',
    filemode='a'
)
logger = logging.getLogger("ConfiguracaoSistema")

@dataclass
class ConfiguracaoMercado:
    """Classse de dados para parâmetros de mercado brasileiro (B3)."""
    moeda_base: str = "BRL"
    taxa_livre_risco_padrao: float = 0.1175  # Selic projetada média
    dias_uteis_ano: int = 252
    tickers_monitorados: List[str] = field(default_factory=lambda: [
        "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "BBAS3.SA",
        "ABEV3.SA", "WEGE3.SA", "RENT3.SA", "SUZB3.SA", "JBSS3.SA"
    ])
    horizonte_previsao_anos: List[int] = field(default_factory=lambda: [2026, 2027, 2028])

@dataclass
class HiperparametrosIA:
    """Hiperparâmetros globais para modelos de Deep Learning."""
    epochs_lstm: int = 150
    batch_size: int = 64
    learning_rate: float = 0.001
    dropout_rate: float = 0.2
    camadas_ocultas_transformer: int = 6
    cabecas_atencao: int = 8

class GerenciadorConfiguracao:
    """
    Singleton para gerenciar o estado global da configuração e persistência.
    """
    _instancia = None

    def __new__(cls):
        if cls._instancia is None:
            cls._instancia = super(GerenciadorConfiguracao, cls).__new__(cls)
            cls._instancia._inicializar()
        return cls._instancia

    def _inicializar(self):
        """Inicializa configurações padrão."""
        self.mercado = ConfiguracaoMercado()
        self.ia = HiperparametrosIA()
        self.caminhos = {
            "dados_brutos": os.path.join(os.getcwd(), "dados", "brutos"),
            "dados_processados": os.path.join(os.getcwd(), "dados", "processados"),
            "modelos_salvos": os.path.join(os.getcwd(), "modelos", "serializados"),
            "relatorios": os.path.join(os.getcwd(), "visualizacao", "relatorios")
        }
        self._criar_diretorios()
        logger.info("Configurações iniciais carregadas com sucesso.")

    def _criar_diretorios(self):
        """Garante que diretórios críticos existam."""
        for chave, caminho in self.caminhos.items():
            try:
                os.makedirs(caminho, exist_ok=True)
                logger.debug(f"Diretório verificado: {caminho}")
            except OSError as e:
                logger.error(f"Falha ao criar diretório {caminho}: {e}")
                raise

    def exportar_config_json(self, arquivo_destino: str = "config_dump.json"):
        """Exporta configuração atual para reproduzibilidade."""
        config_dict = {
            "mercado": self.mercado.__dict__,
            "ia": self.ia.__dict__,
            "caminhos": self.caminhos,
            "timestamp": datetime.now().isoformat()
        }
        try:
            with open(arquivo_destino, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=4)
            logger.info(f"Configurações exportadas para {arquivo_destino}")
        except IOError as e:
            logger.error(f"Erro na exportação de config: {e}")

    def carregar_config_json(self, arquivo_origem: str):
        """Carrega configuração de arquivo JSON externo."""
        if not os.path.exists(arquivo_origem):
            logger.warning(f"Arquivo {arquivo_origem} não encontrado.")
            return

        try:
            with open(arquivo_origem, 'r', encoding='utf-8') as f:
                dados = json.load(f)
                # Atualização simplificada - num caso real faria merge recursivo
                if "mercado" in dados:
                    self.mercado = ConfiguracaoMercado(**dados["mercado"])
                if "ia" in dados:
                    self.ia = HiperparametrosIA(**dados["ia"])
            logger.info(f"Configurações carregadas de {arquivo_origem}")
        except Exception as e:
            logger.error(f"Erro ao carregar config: {e}")

# Instância global
CONFIG = GerenciadorConfiguracao()

# Constantes Matemáticas Estendidas
EPSILON_DOUBLE = 1e-15  # Precisão de ponto flutuante
PI_PRECISAO = 3.14159265358979323846
CONSTANTE_EULER = 0.57721566490153286060

def obter_info_sistema():
    """Retorna metadados do sistema para cabeçalhos de relatórios."""
    return {
        "Sistema": "Super Modelo Black-Scholes Avançado",
        "Autor": "Luiz Tiago Wilcke, Estudante de Estatística Unisociesc",
        "Modo": "Produção/Pesquisa",
        "Versão Python Suportada": "3.10+",
        "Ano Base": 2026
    }
