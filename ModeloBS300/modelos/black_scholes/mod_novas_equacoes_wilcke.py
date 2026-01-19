"""
Módulo de Novas Equações de Wilcke (Modelo de Difusão Híbrida de Sentimento)
============================================================================

Este módulo apresenta a inovação central do projeto: o "Modelo de Difusão Wilcke".
Ele estende a equação clássica de Black-Scholes incorporando um termo estocástico
adicional derivado do "Sentimento de Mercado" (S_t), modelado via Processo de
Ornstein-Uhlenbeck acoplado ao preço do ativo.

Dinâmica Proposta (Wilcke Equations):
1. dS_t = mu*S_t*dt + sigma(V_t)*S_t*dW1_t + alpha*M_t*S_t*dt
2. dM_t = kappa*(theta - M_t)*dt + xi*dW2_t (Sentimento)

Autor: Luiz Tiago Wilcke, Estudante de Estatística
Data: 2026-01-19
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Callable
import logging

logger = logging.getLogger("NovasEquacoesWilcke")

@dataclass
class ParametrosWilcke:
    """Parâmetros do novo modelo estendido."""
    mu: float      # Drift original
    sigma_base: float # Volatilidade base
    kappa: float   # Velocidade reversão sentimento
    theta: float   # Sentimento médio (neutro=0)
    xi: float      # Volatilidade do sentimento
    alpha: float   # Coeficiente de impacto do sentimento no preço
    rho: float     # Correlação entre Preço e Sentimento

class ModeloDifusaoWilcke:
    """
    Implementa a simulação numérica e precificação sob a nova dinâmica proposta.
    """
    
    def __init__(self, semente: int = 2026):
        self.rng = np.random.default_rng(semente)

    def simular_dinamica_acoplada(
        self,
        params: ParametrosWilcke,
        S0: float,
        M0: float, # Sentimento inicial (-1 a +1 tipicamente)
        T: float,
        num_passos: int,
        num_simulacoes: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simula o sistema de equações diferenciais estocásticas (SDEs) acopladas.
        Usando esquema de Euler-Maruyama modificado.
        """
        dt = T / num_passos
        sqrt_dt = np.sqrt(dt)
        
        # Preços e Sentimentos
        sim_S = np.zeros((num_passos + 1, num_simulacoes))
        sim_M = np.zeros((num_passos + 1, num_simulacoes))
        
        sim_S[0] = S0
        sim_M[0] = M0
        
        # Gerar Brownianos Correlacionados
        # Z1, Z2 ~ N(0,1)
        z1 = self.rng.standard_normal((num_passos, num_simulacoes))
        z2_raw = self.rng.standard_normal((num_passos, num_simulacoes))
        z2 = params.rho * z1 + np.sqrt(1 - params.rho**2) * z2_raw
        
        for t in range(num_passos):
            S_t = sim_S[t]
            M_t = sim_M[t]
            
            # Dinâmica do Sentimento (Ornstein-Uhlenbeck)
            # Tende a retornar a theta (neutralidade)
            dM = params.kappa * (params.theta - M_t) * dt + params.xi * sqrt_dt * z2[t]
            M_next = M_t + dM
            
            # Dinâmica do Preço (Wilcke Extended)
            # O drift é ajustado pelo sentimento: (mu + alpha*M_t)
            # Sentimento positivo impulsiona preço acima do drift fundamental
            drift_efetivo = params.mu + params.alpha * M_t
            
            # Discretização geométrica para garantir positividade
            dS_log = (drift_efetivo - 0.5 * params.sigma_base**2) * dt + \
                     params.sigma_base * sqrt_dt * z1[t]
            
            S_next = S_t * np.exp(dS_log)
            
            sim_S[t+1] = S_next
            sim_M[t+1] = M_next
            
        return sim_S, sim_M

    def precificar_opcao_wilcke(
        self,
        params: ParametrosWilcke,
        S0: float,
        K: float,
        T: float,
        tipo: str = 'call',
        num_simulacoes: int = 10000
    ) -> float:
        """
        Precifica opções usando o novo modelo via Monte Carlo.
        A não-linearidade do sentimento impede solução analítica fechada simples.
        """
        sim_S, _ = self.simular_dinamica_acoplada(
            params, S0, M0=0.0, T=T, 
            num_passos=int(T*252), num_simulacoes=num_simulacoes
        )
        
        S_T = sim_S[-1]
        
        # Desconto a taxa livre de risco ajustada (assumindo risk-neutral)
        # Aqui simplificamos usando mu como r risk-neutral no params
        fator_desconto = np.exp(-params.mu * T)
        
        if tipo == 'call':
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)
            
        preco = fator_desconto * np.mean(payoffs)
        erro_padrao = np.std(payoffs) / np.sqrt(num_simulacoes)
        
        logger.info(f"Preço Wilcke ({tipo}): {preco:.4f} +/- {erro_padrao:.4f}")
        return preco

    def calibrar_sentimento(self, retornos_historicos: np.ndarray) -> float:
        """
        Tenta estimar o estado atual do 'fator alfa' (sentimento) 
        baseado na divergência entre preço e fundamentos recentes.
        Função esqueleto.
        """
        # Ex: Sentimento = Retorno Realizado - Retorno Esperado (CAPM)
        # Retorna valor entre -1 e 1 normalizado
        return 0.0 # Placeholder

def comparativo_modelos():
    print("--- Inovação: Modelo de Difusão Wilcke ---")
    
    # Parâmetros de Mercado Turbulento
    params = ParametrosWilcke(
        mu=0.1175,       # Selic
        sigma_base=0.25,
        kappa=2.0,       # Reversão rápida de sentimento
        theta=0.0,       # Neutro a longo prazo
        xi=0.5,          # Sentimento volátil
        alpha=0.3,       # Forte impacto do sentimento no preço
        rho=-0.4         # Efeito alavancagem (Queda gera pânico/medo)
    )
    
    modelo = ModeloDifusaoWilcke()
    
    S0 = 100.0
    K = 105.0
    T = 1.0
    
    # 1. Preço Wilcke (Com sentimento neutro inicial)
    preco_wilcke = modelo.precificar_opcao_wilcke(params, S0, K, T, 'call')
    
    # 2. Preço BS Clássico (Sem alpha/sentimento)
    from scipy.stats import norm
    d1 = (np.log(S0/K) + (params.mu + 0.5*params.sigma_base**2)*T)/(params.sigma_base*np.sqrt(T))
    d2 = d1 - params.sigma_base*np.sqrt(T)
    preco_bs = S0*norm.cdf(d1) - K*np.exp(-params.mu*T)*norm.cdf(d2)
    
    print(f"\nPreço Modelo Wilcke: {preco_wilcke:.4f}")
    print(f"Preço Black-Scholes: {preco_bs:.4f}")
    print(f"Prêmio de Risco do Sentimento: {preco_wilcke - preco_bs:.4f}")
    
    # Simulação Visual
    S, M = modelo.simular_dinamica_acoplada(params, S0, 0, 1.0, 252, 5)
    print(f"\nCaminhos Simulados (Sentimento Final Médio): {np.mean(M[-1]):.4f}")

if __name__ == "__main__":
    comparativo_modelos()
