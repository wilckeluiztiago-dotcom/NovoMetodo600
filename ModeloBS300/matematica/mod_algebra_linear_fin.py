"""
Módulo de Álgebra Linear Financeira
===================================

Este módulo fornece implementações otimizadas e numericamente estáveis de operações
matriciais essenciais para finanças quantitativas, como decomposição de Cholesky,
análise de componentes principais (PCA) para curvas de juros e limpeza de matrizes
de covariância ruidosas (Random Matrix Theory).

Autor: Luiz Tiago Wilcke
Data: 2026-01-19
"""

import numpy as np
from scipy import linalg
from dataclasses import dataclass
from typing import Tuple, List, Optional
import logging

# Configuração de Logger
logger = logging.getLogger("AlgebraLinearFin")

class AlgebraFinanceiraAvancada:
    """
    Toolkit de operações matriciais avançadas para otimização de portfólio e modelagem de risco.
    """

    @staticmethod
    def decomposicao_cholesky_estavel(matriz_cov: np.ndarray) -> np.ndarray:
        """
        Executa a Decomposição de Cholesky (L * L.T = A).
        Implementa fallback para matrizes próximas de singulares (não positivas definidas)
        adicionando um pequeno jitter à diagonal (regularização).
        
        Args:
            matriz_cov: Matriz hermitiana positiva definida (n x n).
            
        Returns:
            Matriz triangular inferior L.
        """
        try:
            return np.linalg.cholesky(matriz_cov)
        except np.linalg.LinAlgError:
            logger.warning("Matriz não é positiva definida. Aplicando correção espectral.")
            return AlgebraFinanceiraAvancada._corrigir_matriz_nao_pd(matriz_cov)

    @staticmethod
    def _corrigir_matriz_nao_pd(matriz: np.ndarray) -> np.ndarray:
        """Tenta corrigir matriz não positiva-definida reconstrutindo via autovalores."""
        val_prop, vec_prop = np.linalg.eigh(matriz)
        # Força autovalores a serem levemente positivos
        val_prop = np.maximum(val_prop, 1e-8)
        matriz_reconstruida = vec_prop @ np.diag(val_prop) @ vec_prop.T
        
        # Garante simetria perfeita
        return np.linalg.cholesky((matriz_reconstruida + matriz_reconstruida.T) / 2)

    @staticmethod
    def analise_componentes_principais(
        dados_retornos: np.ndarray, 
        n_componentes: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Realiza PCA em retornos de ativos para identificar fatores de risco latentes.
        
        Args:
            dados_retornos: Matriz (T x N) de retornos.
            n_componentes: Numero de fatores a extrair. Se None, usa regra de Kaiser.
            
        Returns:
            Tuple(autovalores, autovetores, variancia_explicada)
        """
        cov_mat = np.cov(dados_retornos, rowvar=False)
        autovalores, autovetores = np.linalg.eigh(cov_mat)
        
        # Ordenar decrescente
        idx_ordem = np.argsort(autovalores)[::-1]
        valores_ord = autovalores[idx_ordem]
        vetores_ord = autovetores[:, idx_ordem]
        
        if n_componentes is None:
            # Regra de Kaiser: autovalores > média (ou 1 se for correl)
            n_componentes = np.sum(valores_ord > np.mean(valores_ord))
            
        return valores_ord[:n_componentes], vetores_ord[:, :n_componentes], valores_ord / np.sum(valores_ord)

    @staticmethod
    def simular_bm_correlacionado(
        num_ativos: int, 
        num_passos: int, 
        matriz_correlacao: np.ndarray
    ) -> np.ndarray:
        """
        Simula N movimentos brownianos correlacionados dW_t.
        dX = L * dZ, onde dZ são independentes e L é Cholesky da Correlação.
        
        Returns:
            Matriz (num_passos x num_ativos) de incrementos normais correlacionados.
        """
        L = AlgebraFinanceiraAvancada.decomposicao_cholesky_estavel(matriz_correlacao)
        # Z ~ N(0, 1) independente (num_passos x num_ativos)
        # Precisamos que a correlação seja entre colunas (ativos)
        # X = Z * L.T
        Z = np.random.standard_normal((num_passos, num_ativos))
        X = Z @ L.T
        return X

    @staticmethod
    def denoising_matriz_covariancia(
        matriz_cov_empirica: np.ndarray, 
        t_observacoes: int, 
        n_ativos: int
    ) -> np.ndarray:
        """
        Implementa filtragem de Marcenko-Pastur para remover ruído de matrizes de covariância.
        Essencial para portfólios grandes onde N ~ T.
        
        Teoria: Autovalores abaixo de lambda_max teórico são assumidos como ruído.
        """
        q = t_observacoes / n_ativos
        val_prop, vec_prop = np.linalg.eigh(matriz_cov_empirica)
        
        # Limites de Marcenko-Pastur para ruído branco
        var_estimada = np.mean(np.diag(matriz_cov_empirica)) # Simplificação
        lambda_max = var_estimada * (1 + (1.0/q) + 2*np.sqrt(1.0/q))
        
        # Substitui autovalores de ruído pela média deles
        mask_ruido = val_prop < lambda_max
        val_ruido_medio = np.mean(val_prop[mask_ruido]) if np.any(mask_ruido) else 0.0
        
        val_prop_clean = val_prop.copy()
        val_prop_clean[mask_ruido] = val_ruido_medio
        
        # Reconstrói matriz
        matriz_limpa = vec_prop @ np.diag(val_prop_clean) @ vec_prop.T
        
        # Rescale para manter diagonal original (trace conservation)
        fator_escala = np.trace(matriz_cov_empirica) / np.trace(matriz_limpa)
        matriz_limpa *= fator_escala
        
        return matriz_limpa

def teste_algebra():
    """Validação básica."""
    print("--- Teste de Álgebra Linear Financeira ---")
    alg = AlgebraFinanceiraAvancada()
    
    # Criar matriz quase singular
    A = np.array([[1.0, 0.99, 0.99], [0.99, 1.0, 0.995], [0.99, 0.995, 1.0]])
    print("Matriz Original:")
    print(A)
    
    L = alg.decomposicao_cholesky_estavel(A)
    print("\nDecomposição Cholesky (L):")
    print(L)
    
    # Teste de reconstrução
    rec = L @ L.T
    erro = np.max(np.abs(A - rec))
    print(f"\nErro de reconstrução: {erro:.10f}")
    
    # Teste de Simulação Correlacionada
    sims = alg.simular_bm_correlacionado(3, 10000, A)
    corr_empirica = np.corrcoef(sims.T)
    print("\nCorrelação Empírica da Simulação:")
    print(corr_empirica)

if __name__ == "__main__":
    teste_algebra()
