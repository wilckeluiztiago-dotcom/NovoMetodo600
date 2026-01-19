"""
Módulo de Séries Temporais Clássicas (ARIMA/SARIMA)
===================================================

Implementa frameworks para modelagem linear de séries temporais com seleção automática
de ordens (Auto-ARIMA). Essencial para capturar tendências lineares e sazonalidades
antes do processamento por redes neurais (abordagem híbrida).

Autor: Luiz Tiago Wilcke
Data: 2026-01-19
"""

import pmdarima as pm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger("SeriesTemporaisClassicas")

@dataclass
class ResultadoModelagem:
    modelo: any
    aic: float
    bic: float
    ordem_arima: Tuple[int, int, int]
    ordem_sazonal: Tuple[int, int, int, int]
    residuos: np.ndarray

class ModeladorSeriesTemporais:
    """Wrapper para PMDARIMA e Statsmodels focado em automação."""

    def __init__(self):
        pass

    def verificar_estacionariedade(self, serie: pd.Series) -> bool:
        """
        Executa testes ADF (Augmented Dickey-Fuller) e KPSS para verificar estacionariedade.
        Retorna True se estacionária.
        """
        # ADF: H0 = Não Estacionária (tem raiz unitária)
        adf_result = adfuller(serie.dropna())
        p_value_adf = adf_result[1]
        is_stationary_adf = p_value_adf < 0.05
        
        # KPSS: H0 = Estacionária
        kpss_result = kpss(serie.dropna(), regression='c', nlags="auto")
        p_value_kpss = kpss_result[1]
        is_stationary_kpss = p_value_kpss > 0.05
        
        logger.info(f"Teste ADF p-valor: {p_value_adf:.4f}. Teste KPSS p-valor: {p_value_kpss:.4f}")
        
        if is_stationary_adf and is_stationary_kpss:
            return True
        elif not is_stationary_adf and not is_stationary_kpss:
            logger.info("Série é Não-Estacionária. Diferenciação necessária.")
            return False
        else:
            logger.warning("Conflito entre testes de estacionariedade. Assumindo não-estacionária por precaução.")
            return False

    def ajustar_auto_arima(
        self, 
        serie: pd.Series, 
        sazonal: bool = False, 
        periodo_m: int = 1
    ) -> ResultadoModelagem:
        """
        Busca automaticamente os melhores parâmetros (p,d,q)(P,D,Q)m minimizando AIC.
        Usa stepwise search para eficiência.
        """
        logger.info(f"Iniciando Auto-ARIMA para {serie.name}. Sazonal={sazonal}, m={periodo_m}")
        
        try:
            modelo = pm.auto_arima(
                serie,
                start_p=0, start_q=0,
                max_p=5, max_q=5,
                m=periodo_m if sazonal else 1,
                start_P=0, seasonal=sazonal,
                d=None, D=None, # Deixar o teste determinar ordem de integração
                trace=False,
                error_action='ignore',  
                suppress_warnings=True, 
                stepwise=True
            )
            
            logger.info(f"Melhor modelo encontrado: {modelo.order} {modelo.seasonal_order}")
            
            return ResultadoModelagem(
                modelo=modelo,
                aic=modelo.aic(),
                bic=modelo.bic(),
                ordem_arima=modelo.order,
                ordem_sazonal=modelo.seasonal_order,
                residuos=modelo.resid()
            )
            
        except Exception as e:
            logger.error(f"Erro no Auto-ARIMA: {e}")
            raise

    def prever_sarimax(
        self, 
        modelo_resultado: ResultadoModelagem, 
        passos_futuros: int,
        exogenas_futuras: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gera previsões pontuais e intervalos de confiança.
        """
        # Statsmodels predict
        # O objeto auto_arima do pmdarima tem interface similar ao sklearn, mas w/ statsmodels backend
        
        preds, conf_int = modelo_resultado.modelo.predict(
            n_periods=passos_futuros, 
            X=exogenas_futuras, 
            return_conf_int=True
        )
        
        return np.array(preds), np.array(conf_int)

def teste_sarima():
    print("--- Teste SARIMA Automático ---")
    
    # Gerar série com tendência e sazonalidade
    idx = pd.date_range("2020-01-01", periods=200, freq='M')
    t = np.arange(200)
    # Tendencia linear + Sazonalidade (seno) + Ruido
    serie = 0.5 * t + 10 * np.sin(2 * np.pi * t / 12) + np.random.normal(0, 2, 200)
    ts = pd.Series(serie, index=idx, name="Vendas_Ficticias")
    
    mod = ModeladorSeriesTemporais()
    
    # 1. Teste Estacionariedade
    estavel = mod.verificar_estacionariedade(ts)
    print(f"Série Estacionária? {estavel}")
    
    # 2. Ajuste Auto-ARIMA (Sazonal m=12)
    res = mod.ajustar_auto_arima(ts, sazonal=True, periodo_m=12)
    print(f"Ordem Selecionada: {res.ordem_arima} Sazonal: {res.ordem_sazonal}")
    print(f"AIC: {res.aic:.2f}")
    
    # 3. Previsão
    preds, conf = mod.prever_sarimax(res, passos_futuros=12)
    print("Previsão próximos 12 meses:")
    print(preds)

if __name__ == "__main__":
    teste_sarima()
