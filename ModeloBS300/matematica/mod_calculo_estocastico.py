"""
Módulo de Cálculo Estocástico Avançado
======================================

Este módulo implementa ferramentas fundamentais de cálculo estocástico necessárias
para a modelagem financeira avançada, incluindo Integrais de Itô, Lema de Itô
generalizado e simulação de Movimentos Brownianos e processos de Levy.

O objetivo é fornecer uma base matemática robusta para a precificação de derivativos
e modelagem de risco no mercado brasileiro.

Autor: Luiz Tiago Wilcke
Data: 2026-01-19
"""

import numpy as np
from typing import Tuple, List, Callable, Union, Optional
from dataclasses import dataclass
import scipy.stats as stats
import logging

# Configuração de Logger Local
logger = logging.getLogger("CalculoEstocastico")

# Definição de Tipos para Clareza
ArrayFloat = np.ndarray
FuncaoDifusao = Callable[[float, float], float]  # f(t, S) -> float

@dataclass
class ParametrosProcesso:
    """Estrutura para armazenar parâmetros de processos estocásticos."""
    nome: str
    media_retorno: float  # Drift (mu)
    volatilidade: float   # Sigma
    taxa_saltos: float = 0.0      # Lambda (para Jump Diffusion)
    media_salto: float = 0.0      # Mu_Jump
    desvio_salto: float = 0.0     # Sigma_Jump
    time_step: float = 1.0/252.0  # dt (diário padrão)

class GeradorProcessosEstocasticos:
    """
    Classe responsável por gerar trajetórias de processos estocásticos complexos.
    Utiliza métodos numéricos avançados para discretização.
    """

    def __init__(self, semente: int = 42):
        """
        Inicializa o gerador com uma semente fixa para reprodutibilidade.
        
        Args:
            semente (int): Semente para o gerador de números aleatórios.
        """
        self.rng = np.random.default_rng(semente)
        logger.info(f"Gerador estocástico inicializado com semente {semente}")

    def movimento_browniano_geometrico(
        self,
        params: ParametrosProcesso,
        preco_inicial: float,
        num_passos: int,
        num_simulacoes: int
    ) -> ArrayFloat:
        """
        Gera caminhos para o Movimento Browniano Geométrico (GBM).
        
        Equação: dS_t = mu * S_t * dt + sigma * S_t * dW_t
        Solução: S_t = S_0 * exp((mu - 0.5*sigma^2)*t + sigma*W_t)

        Args:
            params: Parâmetros do processo (drift, vol).
            preco_inicial: Preço inicial do ativo (S0).
            num_passos: Número de passos de tempo na simulação.
            num_simulacoes: Quantidade de caminhos a simular.

        Returns:
            Matriz [num_passos + 1, num_simulacoes] com os preços simulados.
        """
        logger.debug(f"Iniciando simulação GBM para {params.nome}")
        
        try:
            # Pré-alocação de matrizes para eficiência
            dt = params.time_step
            caminhos = np.zeros((num_passos + 1, num_simulacoes))
            caminhos[0] = preco_inicial
            
            # Geração de choques aleatórios (z ~ N(0, 1))
            choques = self.rng.standard_normal((num_passos, num_simulacoes))
            
            # Fator de Drift determinístico
            drift_fator = (params.media_retorno - 0.5 * params.volatilidade**2) * dt
            
            # Fator de Difusão estocástico
            difusao_fator = params.volatilidade * np.sqrt(dt)
            
            # Vetorização completa usando soma cumulativa (Muito mais rápido que for-loop)
            log_retornos = drift_fator + difusao_fator * choques
            caminhos[1:] = preco_inicial * np.exp(np.cumsum(log_retornos, axis=0))
            
            return caminhos
            
        except Exception as e:
            logger.error(f"Erro na simulação GBM: {e}")
            raise

    def jump_diffusion_merton(
        self,
        params: ParametrosProcesso,
        preco_inicial: float,
        num_passos: int,
        num_simulacoes: int
    ) -> ArrayFloat:
        """
        Simula o processo de Difusão com Saltos de Merton.
        Crucial para mercados emergentes como o Brasileiro onde choques são comuns.
        
        Equação: dS_t/S_t = (mu - lambda*k)dt + sigma*dW_t + (Y-1)dq
        
        Args:
            Ver movimento_browniano_geometrico.
        """
        logger.debug(f"Iniciando simulação Jump Diffusion para {params.nome}")
        
        dt = params.time_step
        caminhos = np.zeros((num_passos + 1, num_simulacoes))
        caminhos[0] = preco_inicial
        
        # Parte Contínua (GBM)
        z1 = self.rng.standard_normal((num_passos, num_simulacoes))
        
        # Parte de Salto (Poisson + LogNormal)
        # Poisson determina SE ocorre salto
        # LogNormal determina TAMANHO do salto
        
        # Probabilidade de salto em dt
        prob_salto = params.taxa_saltos * dt
        
        # Máscara de saltos (1 se houve salto, 0 caso contrário)
        ocorrencia_salto = self.rng.poisson(prob_salto, (num_passos, num_simulacoes))
        
        # Magnitude do salto (log-normal)
        # Se Y ~ LogNormal, então log(Y) ~ Normal
        # media_log = mu_jump, desvio_log = sigma_jump
        tamanho_salto_log = self.rng.normal(
            params.media_salto, 
            params.desvio_salto, 
            (num_passos, num_simulacoes)
        )
        
        # Correção de drift para compensar a média dos saltos (mantendo martingale se necessário)
        # k = E[Y - 1] = exp(mu_jump + 0.5*sigma_jump^2) - 1
        k = np.exp(params.media_salto + 0.5 * params.desvio_salto**2) - 1
        drift_correcao = params.taxa_saltos * k
        
        drift_continuo = (params.media_retorno - drift_correcao - 0.5 * params.volatilidade**2) * dt
        difusao_continua = params.volatilidade * np.sqrt(dt) * z1
        
        # Componente de salto acumulado no passo
        # O retorno logarítmico do salto é soma dos logs
        salto_total = ocorrencia_salto * tamanho_salto_log
        
        log_retornos = drift_continuo + difusao_continua + salto_total
        
        caminhos[1:] = preco_inicial * np.exp(np.cumsum(log_retornos, axis=0))
        
        return caminhos

class CalculadoraIto:
    """
    Ferramentas analíticas e numéricas baseadas no Lema de Itô.
    """
    
    @staticmethod
    def integral_ito_numerica(
        funcao_integrando: Callable[[float], float],
        caminho_browniano: ArrayFloat,
        dt: float
    ) -> float:
        """
        Calcula a Integral de Itô numericamente para uma trajetória.
        I = Integral(f(t) dW_t)
        
        Aproximação: Soma(f(t_i-1) * (W_ti - W_ti-1))
        IMPORTANTE: O integrando é avaliado no INÍCIO do intervalo (ponto esquerdo).
        
        Args:
            funcao_integrando: Função f(t) ou f(S_t).
            caminho_browniano: Vetor representando W_t.
            dt: Passo de tempo.
        """
        n = len(caminho_browniano) - 1
        incrementos_w = np.diff(caminho_browniano)
        
        # Avaliação ingênua apenas no tempo, idealmente seria f(S_t)
        # Aqui assumimos f dependendo do índice ou valor anterior
        valores_integrando = np.array([funcao_integrando(val) for val in caminho_browniano[:-1]])
        
        integral = np.sum(valores_integrando * incrementos_w)
        return integral

    @staticmethod
    def variacao_quadratica(caminho: ArrayFloat) -> float:
        """
        Calcula a variação quadrática realizada de um caminho.
        [X, X]_T = lim Soma (X_ti - X_ti-1)^2
        """
        incrementos = np.diff(caminho)
        return np.sum(incrementos**2)

class PonteBrowniana:
    """
    Técnica de redução de variância e interpolação de caminhos.
    Constrói um Movimento Browniano condicionado a W(T) = b.
    Útil para amostragem estratificada.
    """
    
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def gerar_ponte(
        self, 
        t_inicial: float, 
        val_inicial: float,
        t_final: float, 
        val_final: float,
        num_passos: int
    ) -> Tuple[ArrayFloat, ArrayFloat]:
        """
        Gera uma ponte browniana entre (t0, y0) e (T, yT).
        
        B(t) = w(t) - (t/T)w(T) + (t/T)yT + ((T-t)/T)y0 (Simplificado)
        
        Returns:
            Tuple(vetor_tempos, vetor_valores)
        """
        dt = (t_final - t_inicial) / num_passos
        tempos = np.linspace(t_inicial, t_final, num_passos + 1)
        
        # Construção direta via distribuição condicional
        # B(ti+1) ~ N( B(ti) + (val_final - B(ti))/(t_final - ti) * dt, dt * (t_final - ti+1)/(t_final - ti) )
        
        caminho = np.zeros(num_passos + 1)
        caminho[0] = val_inicial
        caminho[-1] = val_final # Garante valor final exato
        
        for i in range(num_passos - 1):
            t_atual = tempos[i]
            t_prox = tempos[i+1]
            dt_atual = t_prox - t_atual
            
            media_condicional = caminho[i] + (val_final - caminho[i]) * (dt_atual / (t_final - t_atual))
            var_condicional = dt_atual * (t_final - t_prox) / (t_final - t_atual)
            if var_condicional < 0: var_condicional = 0 # Correção numérica
            
            caminho[i+1] = self.rng.normal(media_condicional, np.sqrt(var_condicional))
            
        return tempos, caminho

def demonstracao_rapida():
    """Função auxiliar para testar o módulo isoladamente."""
    print("--- Teste do Módulo de Cálculo Estocástico ---")
    gen = GeradorProcessosEstocasticos()
    params = ParametrosProcesso("Petrobras Simulado", 0.15, 0.30, 0.5, -0.05, 0.10)
    
    caminhos = gen.jump_diffusion_merton(params, 30.0, 252, 5)
    print(f"Caminhos gerados shape: {caminhos.shape}")
    print(f"Preço final médio: {np.mean(caminhos[-1, :]):.4f}")
    
    calc = CalculadoraIto()
    var_quad = calc.variacao_quadratica(caminhos[:, 0])
    print(f"Variação Quadrática do caminho 0: {var_quad:.4f}")

if __name__ == "__main__":
    demonstracao_rapida()
