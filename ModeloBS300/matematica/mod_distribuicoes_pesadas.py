"""
Módulo de Distribuições de Caudas Pesadas (Heavy-Tails)
=======================================================

Este módulo implementa distribuições estatísticas não-gaussianas essenciais para
capturar o risco extremo (Cisnes Negros) nos mercados brasileiros. 

Inclui:
- Distribuição Alpha-Stable (Levy Stable)
- T-Student Generalizada Assimétrica (Skew-T)
- Teoria de Valores Extremos (EVT - Generalized Pareto)

Autor: Luiz Tiago Wilcke
Data: 2026-01-19
"""

import numpy as np
from scipy.stats import levy_stable, t, genpareto
from dataclasses import dataclass
from typing import Tuple, List, Optional
import logging

# Logger
logger = logging.getLogger("DistribuicoesPesadas")

@dataclass
class ParametrosEstaveis:
    alpha: float  # Índice de estabilidade (0 < alpha <= 2)
    beta: float   # Assimetria (-1 <= beta <= 1)
    gamma: float  # Escala (> 0)
    delta: float  # Localização (real)

class ModeladorCaudasPesadas:
    """
    Framework para ajuste e simulação de distribuições leptocúrticas.
    """

    def __init__(self):
        pass

    def ajustar_levy_stable(self, dados: np.ndarray) -> ParametrosEstaveis:
        """
        Ajusta uma distribuição estável de Levy aos dados via Máxima Verossimilhança.
        Essa distribuição generaliza a Normal e captura caudas infinitamente variadas.
        
        Args:
            dados: Vetor de retornos logarítmicos.
            
        Returns:
            Objeto ParametrosEstaveis.
        """
        logger.info("Ajustando Distribuição Levy-Stable aos dados...")
        
        # Scipy usa parametrização (alpha, beta, loc, scale)
        # Atenção: Ajuste de ML para Levy é computacionalmente intensivo (FFT)
        params = levy_stable.fit(dados)
        
        p = ParametrosEstaveis(
            alpha=params[0],
            beta=params[1],
            delta=params[2], # loc
            gamma=params[3]  # scale
        )
        
        logger.info(f"Parâmetros ajustados: alpha={p.alpha:.4f} (Cauda), beta={p.beta:.4f} (Assimetria)")
        return p

    def simular_levy_stable(self, params: ParametrosEstaveis, n_amostras: int) -> np.ndarray:
        """Gera números aleatórios seguindo a distribuição estável ajustada."""
        return levy_stable.rvs(
            params.alpha, 
            params.beta, 
            loc=params.delta, 
            scale=params.gamma, 
            size=n_amostras
        )

    def ajustar_t_student_assimetrica(self, dados: np.ndarray) -> Tuple[float, float, float]:
        """
        Ajusta uma distribuição T-Student (robusta a outliers) com graus de liberdade.
        
        Returns:
            (df, loc, scale)
        """
        params = t.fit(dados)
        logger.info(f"T-Student ajustada: GL={params[0]:.2f}")
        return params

    def calcular_var_es_evt(
        self, 
        dados: np.ndarray, 
        confianca: float = 0.99, 
        limiar_u: float = None
    ) -> Tuple[float, float]:
        """
        Calcula Value-at-Risk (VaR) e Expected Shortfall (ES) usando 
        Teoria de Valores Extremos (Peaks Over Threshold - POT).
        
        Args:
           dados: Histórico de retornos (perdas positivas).
           confianca: Nível de confiança (ex: 0.99).
           limiar_u: Threshold para cauda. Se None, usa percentil 90.
           
        Returns:
            (VaR, ES)
        """
        # Trabalha com perdas (inverte sinal se forem retornos)
        perdas = -dados 
        
        if limiar_u is None:
            limiar_u = np.percentile(perdas, 90)
            
        excessos = perdas[perdas > limiar_u] - limiar_u
        n_total = len(perdas)
        n_excessos = len(excessos)
        
        if n_excessos < 10:
            logger.warning("Poucos dados na cauda para EVT confiável.")
            return np.percentile(perdas, confianca*100), np.mean(perdas[perdas > np.percentile(perdas, confianca*100)])

        # Ajuste da GPD (Generalized Pareto Distribution)
        # shape (xi), loc (0), scale (sigma)
        xi, loc, sigma = genpareto.fit(excessos, floc=0)
        
        # Fórmulas analíticas EVT para VaR e ES
        # VaR = u + (sigma/xi) * [ ( (n/Nu) * (1-p) )^(-xi) - 1 ]
        term1 = (n_total / n_excessos) * (1 - confianca)
        var_evt = limiar_u + (sigma / xi) * (np.power(term1, -xi) - 1)
        
        # ES = (VaR + sigma - xi*u) / (1 - xi)
        es_evt = (var_evt + sigma - xi * limiar_u) / (1 - xi)
        
        return var_evt, es_evt

def teste_distribuicoes():
    print("--- Teste de Distribuições de Cauda Pesada ---")
    
    # Gerar dados sintéticos com caudas pesadas (T-Student df=3)
    np.random.seed(42)
    dados = t.rvs(df=3, size=2000) * 0.02
    
    mod = ModeladorCaudasPesadas()
    
    # Teste Levy
    params_levy = mod.ajustar_levy_stable(dados)
    print(f"Levy Alpha (esperado < 2): {params_levy.alpha:.4f}")
    
    # Teste EVT
    var, es = mod.calcular_var_es_evt(dados, confianca=0.99)
    print(f"VaR 99% (EVT): {var:.4f}")
    print(f"ES 99% (EVT): {es:.4f}")

if __name__ == "__main__":
    teste_distribuicoes()
