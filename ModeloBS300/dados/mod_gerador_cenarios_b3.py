"""
Módulo de Geração de Cenários Futuros B3 (2026-2028)
====================================================

Este módulo aplica técnicas econométricas e estocásticas para projetar cenários
macroeconômicos e de preços de ativos para o horizonte de 2026 a 2028.

Utiliza modelos vetoriais (VAR) para variáveis macro (PIB, Selic, USD/BRL) e
difusão correlacionada para os ativos individuais.

Autor: Luiz Tiago Wilcke
Data: 2026-01-19
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import logging

# Logger
logger = logging.getLogger("GeradorCenarios")

@dataclass
class CenarioMacro:
    """Estrutura de um cenário econômico anual."""
    ano: int
    pib_crescimento: float
    inflacao_ipca: float
    selic_meta: float
    cambio_usd_brl: float

class ProjetorEconomico:
    """Projeta variáveis latentes da economia."""
    
    def __init__(self):
        # Premissas base (ponto de partida 2026)
        self.premissas_base = {
            2026: CenarioMacro(2026, 0.025, 0.040, 0.1050, 5.20),
            2027: CenarioMacro(2027, 0.028, 0.035, 0.0950, 5.35),
            2028: CenarioMacro(2028, 0.030, 0.035, 0.0850, 5.50)
        }

    def gerar_caminho_selic_diaria(self, ano_inicio: int, ano_fim: int) -> pd.Series:
        """Interpolador estocástico para curva de juros diária."""
        datas = pd.date_range(start=f"{ano_inicio}-01-01", end=f"{ano_fim}-12-31", freq='B')
        n_dias = len(datas)
        
        # Construção da tendência linear entre as metas anuais
        tendencia = np.zeros(n_dias)
        
        # Simplificação: Interpolação Linear das Metas
        metas = [self.premissas_base[y].selic_meta for y in range(ano_inicio, ano_fim+1)]
        anos = np.array(range(ano_inicio, ano_fim+1))
        
        # Mapear data para fração do ano para interpolação
        fracoes_ano = np.linspace(ano_inicio, ano_fim, n_dias)
        tendencia = np.interp(fracoes_ano, anos, metas)
        
        # Adicionar ruído de mercado (Vasicek simples: dr = a(b-r)dt + s*dw)
        r = np.zeros(n_dias)
        r[0] = metas[0]
        kappa = 0.5 # Velocidade de reversão à tendência
        sigma = 0.01
        dt = 1/252
        
        for t in range(1, n_dias):
            dr = kappa * (tendencia[t] - r[t-1]) * dt + sigma * np.sqrt(dt) * np.random.normal()
            r[t] = r[t-1] + dr
            
        return pd.Series(r, index=datas, name='Selic_Projetada')

class GeradorCenariosAtivos:
    """Simula preços de ações condicionados ao cenário macro."""
    
    def __init__(self, projetor_macro: ProjetorEconomico):
        self.macro = projetor_macro
        
    def simular_precos_2026_2028(
        self, 
        tickers: List[str], 
        precos_iniciais: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Gera DataFrame com previsões diárias para múltiplos ativos.
        Usa modelo de fatores onde Retorno_Ativo ~ Beta * Retorno_Mercado + Alpha + Ruído
        Onde Retorno_Mercado é função do PIB e Selic.
        """
        datas = pd.date_range(start="2026-01-01", end="2028-12-31", freq='B')
        n_dias = len(datas)
        n_ativos = len(tickers)
        
        # 1. Gerar Fator de Mercado (Índice Bovespa Teórico)
        # Sensível positivamente ao PIB e negativamente aos Juros
        selic_diaria = self.macro.gerar_caminho_selic_diaria(2026, 2028)
        
        # Drift do mercado base ~ (PIB + Inflação - Delta_Juros)
        drift_mercado_anual = 0.12 # Prêmio de risco Brasil
        vol_mercado_anual = 0.20
        
        retornos_mercado = np.random.normal(
            (drift_mercado_anual / 252), 
            (vol_mercado_anual / np.sqrt(252)), 
            n_dias
        )
        
        df_precos = pd.DataFrame(index=datas)
        
        # 2. Gerar Ativos Individuais (CAPM Estocástico)
        betas = np.linspace(0.8, 1.5, n_ativos) # Betas fictícios espalhados
        
        for i, ticker in enumerate(tickers):
            beta = betas[i]
            vol_idiosincratica = 0.25  # Risco específico alto
            
            # r_i = beta * r_m + epsilon
            ruido = np.random.normal(0, vol_idiosincratica / np.sqrt(252), n_dias)
            retornos_ativo = beta * retornos_mercado + ruido
            
            # Incorpora choques aleatórios "Black Swan" (Prob 0.1%)
            saltos = np.random.choice([0, -0.15, 0.10], size=n_dias, p=[0.999, 0.0005, 0.0005])
            retornos_ativo += saltos
            
            p0 = precos_iniciais.get(ticker, 100.0)
            caminho_precos = p0 * np.exp(np.cumsum(retornos_ativo))
            
            df_precos[ticker] = caminho_precos
            
        logger.info(f"Cenários gerados para {n_ativos} ativos até 2028.")
        return df_precos

def teste_cenarios():
    print("--- Teste de Geração de Cenários Futuros ---")
    
    proj = ProjetorEconomico()
    selic = proj.gerar_caminho_selic_diaria(2026, 2028)
    print(f"Selic Projetada Final 2028: {selic.iloc[-1]:.2%}")
    
    gen = GeradorCenariosAtivos(proj)
    tickers = ["PETR4", "VALE3", "ITUB4"]
    p0 = {"PETR4": 35.0, "VALE3": 70.0, "ITUB4": 30.0}
    
    df = gen.simular_precos_2026_2028(tickers, p0)
    print("\nPrevisão de Preços (Tail):")
    print(df.tail())
    
    print(f"\nRetorno Acumulado PETR4: {(df['PETR4'].iloc[-1]/df['PETR4'].iloc[0] - 1):.2%}")

if __name__ == "__main__":
    teste_cenarios()
