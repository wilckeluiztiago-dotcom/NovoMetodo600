"""
Super Modelo Black-Scholes - Sistema de Previsão Financeira Avançada
====================================================================

Script principal de orquestração (Main).
Integra:
1. Geração de Cenários Macroeconômicos (2026-2028)
2. Modelagem Híbrida (Wilcke Diffusion + Deep Learning SOTA)
3. Precificação de Derivativos e Cálculo de Risco
4. Visualização 3D Interativa

Autor: Luiz Tiago Wilcke, Estudante de Estatística Unisociesc
Data: 2026-01-19
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Hack para path local
sys.path.append(os.getcwd())

# Importação dos Módulos do Sistema
from utilidades.configuracoes import CONFIG, obter_info_sistema
from dados.mod_gerador_cenarios_b3 import ProjetorEconomico, GeradorCenariosAtivos
from dados.mod_limpeza_tratamento import EngenhariaFeatures
from modelos.black_scholes.mod_novas_equacoes_wilcke import ModeloDifusaoWilcke, ParametrosWilcke
from modelos.black_scholes.mod_bs_classico_avancado import PrecificadorBlackScholes
from modelos.series_temporais.mod_arima_sarima_autom import ModeladorSeriesTemporais

# Novos Imports de IA (API Atualizada)
from modelos.ia.mod_rede_neural_lstm import (
    TreinadorDeepLearning, ConfigRedeNeural, FactoryDados, ScalerFinanceiro
)

from visualizacao.mod_graficos_interativos_3d import VisualizadorFinanceiro

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MainOrquestrador")

def main():
    logger.info("==========================================================")
    logger.info("   INICIANDO SUPER MODELO FINANCEIRO (WILCKE SYSTEM)      ")
    logger.info("==========================================================")
    
    info = obter_info_sistema()
    logger.info(f"Sistema: {info['Sistema']} | Autor: {info['Autor']}")
    
    # -------------------------------------------------------------------------
    # 1. Geração de Cenários Futuros (Camada de Dados)
    # -------------------------------------------------------------------------
    logger.info("\n[FASE 1] Gerando Cenários Macroeconômicos 2026-2028...")
    
    projetor_macro = ProjetorEconomico()
    selic_futura = projetor_macro.gerar_caminho_selic_diaria(2026, 2028)
    
    gerador_ativos = GeradorCenariosAtivos(projetor_macro)
    
    # Portfólio de Teste
    carteira_inicial = {"PETR4": 38.50, "VALE3": 72.00, "WEGE3": 40.00}
    df_cenarios = gerador_ativos.simular_precos_2026_2028(list(carteira_inicial.keys()), carteira_inicial)
    
    logger.info(f"Cenários gerados! Tamanho da base: {df_cenarios.shape}. Término em: {df_cenarios.index[-1]}")
    
    # -------------------------------------------------------------------------
    # 2. Engenharia de Features e IA (Camada de Inteligência)
    # -------------------------------------------------------------------------
    logger.info("\n[FASE 2] Treinando Rede Neural SOTA (LSTM+Attention)...")
    
    # Prepara dados para a IA (usando PETR4 como exemplo piloto)
    series_petr = df_cenarios['PETR4']
    
    # Criar DataFrame com apenas a coluna target para o pipeline simplificado SOTA
    # Observação: O módulo SOTA novo suporta multivariável mas FactoryDados simplifica para univariado
    df_petr_simple = pd.DataFrame({'Close': series_petr})
    
    # Configuração da Rede Neural Avançada
    config_ia = ConfigRedeNeural(
        nome_modelo="PETR4_Predictor_SOTA",
        arquitetura="LSTM", # Ou "LSTM_Attention"
        input_dim=1,
        hidden_dim=64, # Aumentar para produção real
        epochs=10,    # Rápido para demo
        batch_size=32,
        window_size=30
    )
    
    # Preparação Automática de Dados
    data_pack = FactoryDados.preparar_dataloaders(
        df_petr_simple, 
        target_col='Close', 
        window=config_ia.window_size,
        batch_size=config_ia.batch_size
    )
    
    # Treinamento
    treinador = TreinadorDeepLearning(config_ia, data_pack['scaler'])
    metrics = treinador.treinar_modelo(data_pack['train_loader'], data_pack['val_loader'])
    
    # Previsão Recursiva (Fora da amostra simulada - Extrapolação)
    logger.info("Gerando projeção futura via IA...")
    last_window = data_pack['target_data'][-config_ia.window_size:] 
    preds_ia = treinador.prever_recursivo(last_window, n_passos=50)
    
    # Criar índices de data para os futuros
    datas_futuras = pd.date_range(series_petr.index[-1], periods=51, freq='B')[1:]
    serie_ia = pd.Series(preds_ia, index=datas_futuras)
    
    # -------------------------------------------------------------------------
    # 3. Modelagem Estatística Clássica (Benchmark)
    # -------------------------------------------------------------------------
    logger.info("\n[FASE 3] Executando Auto-ARIMA Benchmark...")
    mod_ts = ModeladorSeriesTemporais()
    
    # Ajuste (usando amostra menor para rapidez)
    res_arima = mod_ts.ajustar_auto_arima(series_petr.iloc[:200], sazonal=False)
    preds_arima_raw, _ = mod_ts.prever_sarimax(res_arima, passos_futuros=50)
    
    # -------------------------------------------------------------------------
    # 4. Precificação com Novas Equações de Wilcke (Camada Quant)
    # -------------------------------------------------------------------------
    logger.info("\n[FASE 4] Precificação de Derivativos com Equações de Wilcke...")
    
    # Parâmetros calibrados (fictícios)
    params_wilcke = ParametrosWilcke(
        mu=selic_futura.mean(),
        sigma_base=0.30,
        kappa=1.5,
        theta=0.0,
        xi=0.4,
        alpha=0.25,
        rho=-0.3
    )
    
    modelo_wilcke = ModeloDifusaoWilcke()
    preco_opcao_exotica = modelo_wilcke.precificar_opcao_wilcke(
        params_wilcke, 
        S0=carteira_inicial['PETR4'], 
        K=40.00, 
        T=1.0, 
        tipo='call'
    )
    
    logger.info(f"Preço Opção Call 1y (Wilcke Model): R$ {preco_opcao_exotica:.8f}")
    
    # Comparativo com BS Clássico
    bs_calc = PrecificadorBlackScholes()
    res_bs = bs_calc.precificar_completo(carteira_inicial['PETR4'], 40.00, 1.0, selic_futura.mean(), 0.30)
    logger.info(f"Preço Opção Call 1y (Black-Scholes): R$ {res_bs.preco:.8f}")
    
    # -------------------------------------------------------------------------
    # 5. Visualização e Entrega
    # -------------------------------------------------------------------------
    logger.info("\n[FASE 5] Gerando Relatórios Visuais...")
    viz = VisualizadorFinanceiro()
    
    # Gráfico Fan Chart (IA vs Histórico Recente)
    # Pega apenas o final do histórico para o plot não ficar gigante
    fim_hist = series_petr.iloc[-100:]
    
    fig_fan = viz.plotar_previsao_fan_chart(
        historico=fim_hist, 
        previsao_media=serie_ia, 
        intervalos_confianca=[
            (serie_ia*0.9, serie_ia*1.1),
            (serie_ia*0.8, serie_ia*1.2)
        ],
        titulo="Projeção PETR4 (2026-2028) - Modelo Híbrido SOTA - Autor: Luiz Tiago Wilcke"
    )
    
    if not os.path.exists("visualizacao/relatorios"):
        os.makedirs("visualizacao/relatorios")
        
    fig_fan.savefig("visualizacao/relatorios/projecao_petr4_fan.png")
    
    logger.info("Processo concluído com sucesso. Resultados salvos em visualizacao/relatorios.")

if __name__ == "__main__":
    main()
