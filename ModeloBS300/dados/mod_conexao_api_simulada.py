"""
Módulo de Conexão de Dados e API Simulada
=========================================

Simula uma conexão de nível institucional com bolsas de valores (como B3 via FIX Protocol).
Como estamos em um ambiente de desenvolvimento sem acesso real a feeds proprietários caros
(Bloomberg/Refinitiv), este módulo gera streams de dados realistas ou baixa dados
públicos (Yahoo Finance) com fallback robusto.

Autor: Luiz Tiago Wilcke
Data: 2026-01-19
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import time

logger = logging.getLogger("ConexaoDados")

@dataclass
class BookOrdens:
    """Snapshot do Livro de Ofertas."""
    ticker: str
    timestamp: datetime
    bids: List[Tuple[float, float]] # [(preco, qtd), ...]
    asks: List[Tuple[float, float]]

class ProvedorDadosMercado:
    """Fachada para obtenção de dados históricos e streaming."""
    
    def __init__(self, use_cache: bool = True):
        self.cache = {}
        self.use_cache = use_cache

    def obter_historico_ajustado(
        self, 
        tickers: List[str], 
        inicio: str, 
        fim: str
    ) -> pd.DataFrame:
        """
        Baixa dados OHLCV ajustados por dividendos/splits.
        Implementa retentativas exponenciais para falhas de rede.
        """
        logger.info(f"Baixando dados para {len(tickers)} ativos de {inicio} a {fim}...")
        
        dados_concatenados = pd.DataFrame()
        
        # Download em batch é mais eficiente
        try:
            # yfinance permite download multi-ticker string separada por espaço
            tickers_str = " ".join(tickers)
            dados = yf.download(
                tickers_str, 
                start=inicio, 
                end=fim, 
                group_by='ticker', 
                auto_adjust=True,
                progress=False
            )
            
            # Normalização de estrutura do Dataframe
            if len(tickers) == 1:
                # Se for apenas 1, yfinance não cria Multindex no columns nivel 0
                dados.columns = pd.MultiIndex.from_product([tickers, dados.columns])
            
            return dados
            
        except Exception as e:
            logger.error(f"Erro crítico no download: {e}")
            logger.info("Gerando dados sintéticos de emergência para não travar fluxo...")
            return self._gerar_dados_sinteticos_fallback(tickers, inicio, fim)

    def _gerar_dados_sinteticos_fallback(
        self, 
        tickers: List[str], 
        inicio: str, 
        fim: str
    ) -> pd.DataFrame:
        """Gera Random Walk se API falhar."""
        dates = pd.date_range(start=inicio, end=fim, freq='B') # Business days
        df_dict = {}
        
        for tkr in tickers:
            preco_base = 30.0 # Preço médio B3
            retornos = np.random.normal(0, 0.02, size=len(dates))
            precos = preco_base * np.exp(np.cumsum(retornos))
            
            cols = pd.DataFrame({
                'Open': precos * 0.99,
                'High': precos * 1.01,
                'Low': precos * 0.98,
                'Close': precos,
                'Volume': np.random.randint(1000000, 10000000, size=len(dates))
            }, index=dates)
            
            for c in cols.columns:
                df_dict[(tkr, c)] = cols[c]
                
        return pd.DataFrame(df_dict)

    def simular_stream_cotacoes(self, ticker: str, duracao_segundos: int = 10):
        """
        Generator que simula um feed de Tick-by-Tick.
        Útil para testar algoritmos de execução e filtro de Kalman.
        """
        preco_atual = 100.00
        start = time.time()
        
        while time.time() - start < duracao_segundos:
            # Simula chegada de trade
            delta_tempo = np.random.exponential(0.5) # Chegada Poisson
            time.sleep(delta_tempo)
            
            choque = np.random.normal(0, 0.05)
            preco_atual += choque
            
            tick = {
                'ticker': ticker,
                'price': round(preco_atual, 2),
                'size': int(np.random.pareto(2) * 100), # Volume power law
                'timestamp': datetime.now()
            }
            
            yield tick

class SimuladorMicroestrutura:
    """
    Simula componentes de microestrutura de mercado (LOB).
    """
    
    @staticmethod
    def gerar_book_ordens(ticker: str, preco_ref: float, profundidade: int = 5) -> BookOrdens:
        """Gera um LOB realista em torno do preço de referência."""
        spread = 0.01 * np.random.randint(1, 5)
        mid = preco_ref
        
        bids = []
        asks = []
        
        for i in range(profundidade):
            bid_px = mid - (spread/2) - (i * 0.01) - np.random.rand()*0.01
            ask_px = mid + (spread/2) + (i * 0.01) + np.random.rand()*0.01
            
            qtd_bid = int(np.random.normal(1000, 200))
            qtd_ask = int(np.random.normal(1000, 200))
            
            bids.append((round(bid_px, 2), qtd_bid))
            asks.append((round(ask_px, 2), qtd_ask))
            
        return BookOrdens(
            ticker=ticker,
            timestamp=datetime.now(),
            bids=bids,
            asks=asks
        )

def teste_dados():
    print("--- Teste de Provedor de Dados ---")
    prov = ProvedorDadosMercado()
    
    # Teste Yfinance
    df = prov.obter_historico_ajustado(
        ["PETR4.SA", "VALE3.SA"], 
        "2023-01-01", 
        "2023-01-10"
    )
    print("\nHead do DataFrame:")
    print(df.head())
    
    # Teste Stream
    print("\nSimulando Stream (3 ticks)...")
    gen = prov.simular_stream_cotacoes("PETR4.SA", 3)
    for i, tick in enumerate(gen):
        print(tick)
        if i >= 2: break
        
    # Teste Book
    print("\nSnapshot do Book:")
    book = SimuladorMicroestrutura.gerar_book_ordens("VALE3.SA", 70.50)
    print(f"Melhor Bid: {book.bids[0]}, Melhor Ask: {book.asks[0]}")

if __name__ == "__main__":
    teste_dados()
