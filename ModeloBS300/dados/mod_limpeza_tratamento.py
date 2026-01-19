"""
Módulo de Limpeza, Tratamento e Engenharia de Features (ETL)
============================================================

Implementa pipelines robustos para pré-processamento de séries temporais financeiras.
Inclui tratamento de dados faltantes (imputação), detecção de outliers, winsorização,
cálculo de retornos logarítmicos e geração de indicadores técnicos.

Autor: Luiz Tiago Wilcke
Data: 2026-01-19
"""

import pandas as pd
import numpy as np
from typing import List, Union, Optional
from scipy.stats import zscore
from dataclasses import dataclass
import logging

logger = logging.getLogger("ETL_Financeiro")

class ProcessadorDados:
    """Toolkit para manipulação e saneamento de DataFrames financeiros."""

    @staticmethod
    def calcular_retornos_log(
        precos: Union[pd.Series, pd.DataFrame], 
        periodo: int = 1
    ) -> Union[pd.Series, pd.DataFrame]:
        """Calcula log-retornos contínuos: ln(Pt / Pt-1)."""
        return np.log(precos / precos.shift(periodo)).dropna()

    @staticmethod
    def imputar_dados_faltantes(
        df: pd.DataFrame, 
        metodo: str = 'time'
    ) -> pd.DataFrame:
        """
        Preenche NaNs de forma inteligente.
        'time': Interpolação linear temporal (padrão para séries).
        'ffill': Last Observation Carried Forward.
        """
        df_clean = df.copy()
        if metodo == 'time':
            df_clean = df_clean.interpolate(method='time')
        elif metodo == 'ffill':
            df_clean = df_clean.ffill()
        
        # Se ainda sobrar algo (no início), bfill
        return df_clean.bfill()

    @staticmethod
    def detectar_remover_outliers(
        serie: pd.Series, 
        z_thresh: float = 3.5,
        metodo: str = 'winsorize'
    ) -> pd.Series:
        """
        Trata outliers estatísticos baseados em Z-Score.
        Winsorize: Clampa valores extremos no limiar.
        Drop: Remove os pontos (cuidado com séries temporais).
        """
        z_scores = np.abs(zscore(serie.dropna()))
        mask_outliers = z_scores > z_thresh
        
        if not np.any(mask_outliers):
            return serie
            
        logger.info(f"Detectados {np.sum(mask_outliers)} outliers em {serie.name}")
        
        serie_tratada = serie.copy()
        
        if metodo == 'winsorize':
            limite_sup = serie.mean() + z_thresh * serie.std()
            limite_inf = serie.mean() - z_thresh * serie.std()
            serie_tratada = serie_tratada.clip(lower=limite_inf, upper=limite_sup)
        elif metodo == 'nan':
            # Marca como NaN para imputar depois
            # Requer reindexação correta pois z_scores ignora nan original
            # Simplificação aqui
            pass
            
        return serie_tratada

class EngenhariaFeatures:
    """Geração de indicadores técnicos para modelos de ML."""
    
    @staticmethod
    def adicionar_indicadores_tecnicos(df_ohlc: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona métricas clássicas (RSI, Bollinger, MACD) ao DataFrame.
        Espera colunas: ['Close', 'High', 'Low', 'Volume'] (case insensitive).
        """
        df = df_ohlc.copy()
        # Padronizar nomes
        mapa_colunas = {c: c.capitalize() for c in df.columns}
        df.rename(columns=mapa_colunas, inplace=True)
        
        close = df['Close']
        
        # 1. Medias Moveis Simples e Exponenciais
        df['SMA_20'] = close.rolling(window=20).mean()
        df['EMA_20'] = close.ewm(span=20, adjust=False).mean()
        
        # 2. Bollinger Bands (20, 2)
        std_20 = close.rolling(window=20).std()
        df['BB_Upper'] = df['SMA_20'] + 2 * std_20
        df['BB_Lower'] = df['SMA_20'] - 2 * std_20
        
        # 3. RSI (Relative Strength Index) 14
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # 4. MACD (12, 26, 9)
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 5. Volatilidade Histórica (Janela Móvel 21 dias)
        # Anualizada
        df['Vol_Hist_21d'] = close.pct_change().rolling(window=21).std() * np.sqrt(252)
        
        return df.dropna()

def teste_etl():
    print("--- Teste de ETL e Engenharia de Features ---")
    
    # Gerar dados dummy
    idx = pd.date_range("2023-01-01", periods=100)
    dados = pd.DataFrame({
        'Close': np.cumsum(np.random.randn(100)) + 100,
        'High': np.cumsum(np.random.randn(100)) + 102,
        'Low': np.cumsum(np.random.randn(100)) + 98,
        'Volume': np.random.randint(100, 1000, 100)
    }, index=idx)
    
    # Inserir NaN e Outlier
    dados.iloc[10, 0] = np.nan
    dados.iloc[50, 0] = 5000.0 # Outlier absurdo
    
    proc = ProcessadorDados()
    eng = EngenhariaFeatures()
    
    # 1. Pipeline de Limpeza
    # Remover outlier
    dados['Close'] = proc.detectar_remover_outliers(dados['Close'], metodo='winsorize')
    
    # Imputar NaN
    dados = proc.imputar_dados_faltantes(dados)
    
    # 2. Features
    df_feat = eng.adicionar_indicadores_tecnicos(dados)
    
    print("Colunas Geradas:", df_feat.columns.tolist())
    print("\nÚltimos registros com RSI e MACD:")
    print(df_feat[['Close', 'RSI_14', 'MACD', 'Vol_Hist_21d']].tail())

if __name__ == "__main__":
    teste_etl()
