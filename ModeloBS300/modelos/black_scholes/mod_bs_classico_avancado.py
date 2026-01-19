"""
Módulo Black-Scholes Clássico Avançado (BSM)
============================================

Implementação rigorosa da fórmula de Black-Scholes-Merton (1973) para precificação
de opções Europeias, estendida para calcular não apenas as Gregas básicas (Delta, Gamma, Vega),
mas também sensibilidades de ordem superior (Vanna, Volga, Color, Speed) essenciais
para hedge dinâmico em mercados voláteis.

Autor: Luiz Tiago Wilcke
Data: 2026-01-19
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Dict, Union, Optional
import logging

logger = logging.getLogger("BlackScholesAvancado")

@dataclass
class ResultadoBSM:
    preco: float
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float
    vanna: float  # dDelta/dVol ou dVega/dS
    volga: float  # dVega/dVol (Vomma)
    color: float  # dGamma/dt
    speed: float  # dGamma/dS
    ultima: float # zomma aka dGamma/dVol
    prob_exercicio: float # N(d2)

class PrecificadorBlackScholes:
    """Calculadora analítica para modelo Log-Normal Geométrico."""
    
    def __init__(self, tipo_juros: str = 'continuo'):
        self.tipo_juros = tipo_juros # continuo ou simples

    def calcular_d1_d2(self, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0):
        """Calcula os termos d1 e d2 padronizados."""
        # Evitar divisão por zero e log(0)
        if T <= 0: return 0.0, 0.0
        if sigma <= 0: return np.inf, np.inf
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    def precificar_completo(
        self,
        S: float,         # Preço Spot
        K: float,         # Strike
        T: float,         # Tempo até expiração (anos)
        r: float,         # Taxa livre de risco
        sigma: float,     # Volatilidade implícita
        tipo: str = 'call',
        q: float = 0.0    # Dividend yield continuo
    ) -> ResultadoBSM:
        """
        Retorna preço e todas as gregas sensíveis.
        """
        if T <= 0:
            # Opção expirada
            val_intrinseco = max(S - K, 0.0) if tipo == 'call' else max(K - S, 0.0)
            return ResultadoBSM(val_intrinseco, 0,0,0,0,0,0,0,0,0,0,0)

        d1, d2 = self.calcular_d1_d2(S, K, T, r, sigma, q)
        
        # Cache de PDFs e CDFs para eficiência
        nd1 = norm.cdf(d1)
        nd2 = norm.cdf(d2)
        n_d1 = norm.pdf(d1) # PDF(d1)
        
        n_neg_d1 = norm.cdf(-d1)
        n_neg_d2 = norm.cdf(-d2)
        
        sqrt_T = np.sqrt(T)
        exp_qt = np.exp(-q * T)
        exp_rt = np.exp(-r * T)
        
        # --- Preço ---
        if tipo == 'call':
            preco = S * exp_qt * nd1 - K * exp_rt * nd2
            delta = exp_qt * nd1
            rho = K * T * exp_rt * nd2
            prob_ex = nd2
        else: # put
            preco = K * exp_rt * n_neg_d2 - S * exp_qt * n_neg_d1
            delta = -exp_qt * n_neg_d1
            rho = -K * T * exp_rt * n_neg_d2
            prob_ex = n_neg_d2

        # --- Gregas de 2ª Ordem (Comuns a Call e Put, ajustado por q) ---
        # Gamma: Curvatura (convexidade)
        gamma = (exp_qt * n_d1) / (S * sigma * sqrt_T)
        
        # Vega: Sensibilidade à Volatilidade
        vega = S * exp_qt * n_d1 * sqrt_T
        if vega < 1e-8: vega = 1e-8 # Evitar divs por zero em vol implícita
        
        # Theta: Sensibilidade ao Tempo (Time Decay)
        # Mais complexo, varia entre call/put
        termo_comum_theta = -(S * exp_qt * n_d1 * sigma) / (2 * sqrt_T)
        
        if tipo == 'call':
            theta = termo_comum_theta - r * K * exp_rt * nd2 + q * S * exp_qt * nd1
        else:
            theta = termo_comum_theta + r * K * exp_rt * n_neg_d2 - q * S * exp_qt * n_neg_d1
            
        # --- Gregas de Ordem Superior (Hedge de Vola e Gamma) ---
        
        # Vanna: Sensibilidade do Delta à Volatilidade (dDelta/dVol)
        # Crucial para hedgear Skew de volatilidade
        vanna = -exp_qt * n_d1 * (d2 / sigma)
        
        # Volga (Vomma): Convexidade da Volatilidade (dVega/dVol)
        # Crucial para hedge de longas caudas (fat tails)
        volga = vega * (d1 * d2 / sigma)
        
        # Speed: Taxa de mudança do Gamma (dGamma/dS)
        speed = -(gamma / S) * ((d1 / (sigma * sqrt_T)) + 1)
        
        # Color: Sensibilidade do Gamma ao Tempo (dGamma/dt)
        color = -0.5 * gamma * (1 + (d1 * (2 * (r-q) * T - d2 * sigma * sqrt_T)) / (sigma * sqrt_T)) / T # Aproximado
        
        # Zomma: Sensibilidade do Gamma à Volatilidade
        zomma = gamma * ((d1 * d2 - 1) / sigma)

        return ResultadoBSM(
            preco=preco,
            delta=delta,
            gamma=gamma,
            vega=vega / 100, # Escala tradicional (mudança por 1% vol)
            theta=theta / 365, # Diário
            rho=rho / 100, # Mudança por 1% juros
            vanna=vanna,
            volga=volga / 100,
            color=color,
            speed=speed,
            ultima=zomma,
            prob_exercicio=prob_ex
        )

    def volatilidade_implicita(
        self,
        preco_mercado: float,
        S: float, K: float, T: float, r: float, 
        tipo: str = 'call',
        tol: float = 1e-5,
        max_iter: int = 100
    ) -> float:
        """
        Calcula Vol Implícita usando Newton-Raphson.
        Encontra sigma tal que BS(sigma) = preco_mercado.
        """
        sigma = 0.5 # Chute inicial
        
        for i in range(max_iter):
            res = self.precificar_completo(S, K, T, r, sigma, tipo)
            diff = res.preco - preco_mercado
            
            if abs(diff) < tol:
                return sigma
            
            # Vega é a derivada do preço em relação à sigma
            # Newton: x_new = x_old - f(x)/f'(x)
            # sigma_new = sigma - (Preco(sigma) - Mercado) / Vega(sigma)
            # Vega ajustado de volta para escala unitária
            vega_unitario = res.vega * 100 
            
            if vega_unitario < 1e-8: # Preço insensível à vol (muito ITM/OTM)
                break
                
            sigma = sigma - diff / vega_unitario
            
        logger.warning(f"Volatilidade Implícita não convergiu para {tipo} K={K}")
        return np.nan

def teste_bs():
    print("--- Teste Black-Scholes Avançado ---")
    bs = PrecificadorBlackScholes()
    
    # Exemplo PETR4 ATM Call
    S = 30.00
    K = 30.00
    T = 0.08 # ~1 mês
    r = 0.1175 # Selic
    sigma = 0.35 # 35% vol
    
    res = bs.precificar_completo(S, K, T, r, sigma, tipo='call')
    
    print(f"Preço Call: {res.preco:.4f}")
    print(f"Delta: {res.delta:.4f} (Prob Exercicio ~ {res.prob_exercicio:.4f})")
    print(f"Gamma: {res.gamma:.4f}")
    print(f"Vega: {res.vega:.4f}")
    print(f"Theta: {res.theta:.4f}/dia")
    print(f"Vanna (dDelta/dVol): {res.vanna:.4f}")
    print(f"Volga (Convexidade Vol): {res.volga:.4f}")
    
    # Vol Implícita Reversa
    vol_calc = bs.volatilidade_implicita(res.preco, S, K, T, r, 'call')
    print(f"Vol Implícita Recuperada: {vol_calc:.2%}")

if __name__ == "__main__":
    teste_bs()
