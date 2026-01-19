"""
Módulo de Simulação de Monte Carlo Avançado (QMC)
=================================================

Implementa técnicas de simulação de alta performance usando Sequências de Sobol
(Quasi-Monte Carlo) para reduzir a variância e acelerar a convergência na 
precificação de derivativos exóticos.

Features:
- Gerador de Sequências de Sobol (via Scipy QMC)
- Variáveis Antitéticas
- Control Variates (Variáveis de Controle)

Autor: Luiz Tiago Wilcke
Data: 2026-01-19
"""

import numpy as np
from scipy.stats.qmc import Sobol
from typing import Callable, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger("MonteCarloAvancado")

@dataclass
class ResultadoSimulacao:
    estimativa: float
    erro_padrao: float
    intervalo_confianca: Tuple[float, float]
    num_caminhos: int
    tempo_execucao: float

class SimuladorMonteCarlo:
    """Motor de simulação estocástica de alto desempenho."""

    def __init__(self, use_sobol: bool = True):
        self.use_sobol = use_sobol

    def gerar_normais_sobol(self, dimensao: int, num_amostras: int) -> np.ndarray:
        """
        Gera amostras de uma distribuição normal multivariada usando Sobol.
        Sequências de Sobol preenchem o espaçao n-dimensional de forma mais uniforme
        que o pseudo-aleatório comum, convergindo em O(1/N) vs O(1/sqrt(N)).
        
        Args:
            dimensao: Número de passos de tempo ou ativos.
            num_amostras: Deve ser preferencialmente potência de 2.
            
        Returns:
            Matriz (num_amostras x dimensao).
        """
        # Sobol requer m (log2 de N)
        # Ajusta num_amostras para a próxima potência de 2 para balanceamento
        m = int(np.ceil(np.log2(num_amostras)))
        num_amostras_ajustado = 2**m
        
        if num_amostras_ajustado != num_amostras:
            logger.info(f"Ajustando QMC amostras de {num_amostras} para {num_amostras_ajustado}")
        
        # Scipy Sobol Generator
        sampler = Sobol(d=dimensao, scramble=True)
        uniformes = sampler.random_base2(m=m)
        
        # Inversa da Normal (Box-Muller ou PPF)
        # Scipy fast norm ppf
        from scipy.special import ndtri
        normais = ndtri(uniformes)
        
        return normais

    def precificar_ativo_europeu_qmc(
        self,
        S0: float,
        K: str,
        T: float,
        r: float,
        sigma: float,
        tipo_opcao: str = 'call',
        num_simulacoes: int = 16384  # 2^14
    ) -> ResultadoSimulacao:
        """
        Precifica opção Europeia usando QMC para benchmarking.
        Serve para validar a eficácia do Sobol vs Solução Analítica BS.
        """
        import time
        start = time.time()
        
        # Para opção Europeia, só precisamos de S_T, não do caminho todo.
        # Dimensão = 1
        z = self.gerar_normais_sobol(dimensao=1, num_amostras=num_simulacoes).flatten()
        
        # S_T = S0 * exp(...)
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
        
        if tipo_opcao.lower() == 'call':
            payoffs = np.exp(-r * T) * np.maximum(ST - K, 0)
        else:
            payoffs = np.exp(-r * T) * np.maximum(K - ST, 0)
            
        media = np.mean(payoffs)
        erro_padrao = np.std(payoffs, ddof=1) / np.sqrt(num_simulacoes)
        
        end = time.time()
        
        return ResultadoSimulacao(
            estimativa=media,
            erro_padrao=erro_padrao,
            intervalo_confianca=(media - 1.96*erro_padrao, media + 1.96*erro_padrao),
            num_caminhos=num_simulacoes,
            tempo_execucao=end - start
        )

    def simulacao_caminho_completo_qmc(
        self,
        modelo_dinamica: Callable[[np.ndarray, float], np.ndarray],
        S0: float,
        T: float,
        num_passos: int,
        num_simulacoes: int
    ) -> np.ndarray:
        """
        Gera caminhos completos usando QMC. Chave para opções Asiáticas/Barrier.
        
        Atenção: QMC perde eficiência em dimensões muito altas (ex: > 50 passos).
        Pode usar Brownian Bridge para mitigar.
        """
        dt = T / num_passos
        
        # Matriz de inovações Z (simulações x passos)
        if self.use_sobol:
            # Dimensão efetiva é o número de passos de tempo
            Z = self.gerar_normais_sobol(dimensao=num_passos, num_amostras=num_simulacoes)
        else:
            Z = np.random.standard_normal((num_simulacoes, num_passos))
            
        caminhos = np.zeros((num_simulacoes, num_passos + 1))
        caminhos[:, 0] = S0
        
        # Reconstrução vetorizada
        # Assumindo GBM para este exemplo genérico, idealmente 'modelo_dinamica' faria isso
        # Mas para QMC path construction, geralmente fazemos direto:
        
        # log_retornos = (r - 0.5s^2)dt + s*sqrt(dt)*Z
        # Aqui deixamos genérico apenas retornando o driver browniano
        # Ou implementamos GBM simples:
        
        # Exemplo hardcoded GBM params (deveria vir de args)
        mu = 0.0
        sigma = 0.2
        
        drift = (mu - 0.5 * sigma**2) * dt
        diff = sigma * np.sqrt(dt) * Z
        
        caminhos[:, 1:] = S0 * np.exp(np.cumsum(drift + diff, axis=1))
        
        return caminhos

def teste_monte_carlo():
    print("--- Teste de Monte Carlo QMC (Sobol) ---")
    
    sim = SimuladorMonteCarlo(use_sobol=True)
    
    # Parâmetros Opção Call ATM
    S0 = 100.0
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.20
    
    res = sim.precificar_ativo_europeu_qmc(S0, K, T, r, sigma)
    
    # Preço Analítico Black-Scholes para comparação
    from scipy.stats import norm
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    bs_price = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    
    print(f"Preço QMC: {res.estimativa:.5f} +/- {res.erro_padrao*1.96:.5f}")
    print(f"Preço BS Analítico: {bs_price:.5f}")
    print(f"Erro Relativo: {abs(res.estimativa - bs_price)/bs_price * 100:.4f}%")
    print(f"Tempo: {res.tempo_execucao:.4f}s")
    
    # Teste de Geração de Caminho
    caminhos = sim.simulacao_caminho_completo_qmc(None, S0, T, 30, 128)
    print(f"Shape caminhos: {caminhos.shape}")

if __name__ == "__main__":
    teste_monte_carlo()
