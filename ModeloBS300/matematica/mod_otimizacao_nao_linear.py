"""
Módulo de Otimização Não-Linear para Calibração de Modelos
==========================================================

Este módulo implementa algoritmos de otimização numérica robustos, focados na
calibração de superfícies de volatilidade e ajuste de parâmetros de modelos complexos
(como Heston e Jump Diffusion) onde as funções objetivo são frequentemente não-convexas.

Algoritmos incluídos:
- Levenberg-Marquardt (Diferenciação automática aproximada)
- Simulated Annealing (Para escapar de mínimos locais)
- Newton-Raphson Multivariado

Autor: Luiz Tiago Wilcke
Data: 2026-01-19
"""

import numpy as np
from scipy.optimize import minimize, least_squares
from typing import Callable, List, Tuple, Dict, Any
import logging
from dataclasses import dataclass

# Logger específico
logger = logging.getLogger("OtimizacaoNaoLinear")

# Tipos
Vetor = np.ndarray
FuncaoCusto = Callable[[Vetor], float]
Gradiente = Callable[[Vetor], Vetor]
Hessiana = Callable[[Vetor], np.ndarray]

@dataclass
class ResultadoOtimizacao:
    """Container para resultados de otimização."""
    sucesso: bool
    parametros_otimos: Vetor
    valor_funcao: float
    iteracoes: int
    mensagem: str
    historico_custo: List[float]

class OtimizadorFinanceiro:
    """
    Wrapper de alto nível para técnicas de otimização aplicadas a finanças.
    """
    
    def __init__(self, tolerancia: float = 1e-8, max_iter: int = 2000):
        self.tol = tolerancia
        self.max_iter = max_iter

    def calibrar_levenberg_marquardt(
        self,
        funcao_residuos: Callable[[Vetor], Vetor],
        chute_inicial: Vetor,
        limites: List[Tuple[float, float]] = None
    ) -> ResultadoOtimizacao:
        """
        Algoritmo padrão ouro para ajuste de curvas (Least Squares Não-Linear).
        Excelente para calibrar modelos de volatilidade local/estocástica aos preços de mercado.
        
        Args:
            funcao_residuos: f(params) -> [erro1, erro2, ...]
            chute_inicial: Ponto de partida.
            limites: Bounds [(min, max), ...]
            
        Returns:
            ResultadoOtimizacao completo.
        """
        logger.info("Iniciando calibração via Levenberg-Marquardt...")
        
        # Levenberg-Marquardt puro no scipy não aceita bounds facilmente, 
        # usamos Trust Region Reflective ('trf') que é uma evolução robusta
        # para problemas limitados.
        metodo = 'trf' if limites else 'lm'
        bounds = (-np.inf, np.inf)
        
        if limites:
            # Converter formato de limites lista de tuplas -> tuple de arrays
            lower = [l[0] if l[0] is not None else -np.inf for l in limites]
            upper = [l[1] if l[1] is not None else np.inf for l in limites]
            bounds = (lower, upper)
            
        try:
            res = least_squares(
                funcao_residuos, 
                chute_inicial, 
                method=metodo, 
                bounds=bounds,
                ftol=self.tol,
                xtol=self.tol,
                gtol=self.tol,
                max_nfev=self.max_iter,
                verbose=0
            )
            
            sucesso = res.success
            msg = res.message
            params = res.x
            custo_final = res.cost # 0.5 * sum(residuals**2)
            
            logger.info(f"Otimização concluída. Sucesso: {sucesso}. Custo: {custo_final:.6f}")
            
            return ResultadoOtimizacao(
                sucesso=sucesso,
                parametros_otimos=params,
                valor_funcao=custo_final,
                iteracoes=res.nfev,
                mensagem=msg,
                historico_custo=[] # least_squares não retorna histórico fácil
            )
            
        except Exception as e:
            logger.error(f"Falha crítica na otimização LM: {e}")
            return ResultadoOtimizacao(False, chute_inicial, np.inf, 0, str(e), [])

    def simulated_annealing_adaptativo(
        self,
        funcao_objetivo: FuncaoCusto,
        limites: List[Tuple[float, float]],
        temp_inicial: float = 100.0,
        temp_final: float = 1e-4,
        taxa_resfriamento: float = 0.95
    ) -> ResultadoOtimizacao:
        """
        Implementação customizada de Simulated Annealing (Têmpera Simulada).
        Vital para funções objetivo "rugosas" com muitos mínimos locais, comum
        em calibração de modelos multifatoriais.
        
        Args:
            funcao_objetivo: f(x) -> escalar (custo a minimizar)
        """
        logger.info("Iniciando Simulated Annealing Adaptativo...")
        
        x_atual = np.array([np.random.uniform(l, u) for l, u in limites])
        custo_atual = funcao_objetivo(x_atual)
        
        x_melhor = x_atual.copy()
        custo_melhor = custo_atual
        
        temperatura = temp_inicial
        iteracao = 0
        historico = [custo_melhor]
        
        n_dims = len(limites)
        
        while temperatura > temp_final and iteracao < self.max_iter:
            # Geração de vizinho (perturbação normal escalada pela temp/limites)
            x_novo = x_atual.copy()
            
            for i in range(n_dims):
                amplitude = (limites[i][1] - limites[i][0]) * (temperatura / temp_inicial) * 0.5
                perturbacao = np.random.normal(0, amplitude)
                x_novo[i] = np.clip(x_novo[i] + perturbacao, limites[i][0], limites[i][1])
                
            custo_novo = funcao_objetivo(x_novo)
            
            # Critério de Metropolis
            delta_e = custo_novo - custo_atual
            
            aceito = False
            if delta_e < 0:
                aceito = True
            else:
                prob = np.exp(-delta_e / temperatura)
                if np.random.rand() < prob:
                    aceito = True
            
            if aceito:
                x_atual = x_novo
                custo_atual = custo_novo
                if custo_atual < custo_melhor:
                    x_melhor = x_atual.copy()
                    custo_melhor = custo_atual
            
            temperatura *= taxa_resfriamento
            iteracao += 1
            
            if iteracao % 100 == 0:
                historico.append(custo_melhor)
        
        return ResultadoOtimizacao(
            sucesso=True,
            parametros_otimos=x_melhor,
            valor_funcao=custo_melhor,
            iteracoes=iteracao,
            mensagem="Convergência Simulated Annealing",
            historico_custo=historico
        )

    def newton_raphson_financeiro(
        self,
        funcao_obj: FuncaoCusto,
        gradiente: Gradiente,
        hessiana: Hessiana,
        chute: Vetor
    ) -> ResultadoOtimizacao:
        """
        Método de Newton modificado para otimização irrestrita rápida.
        Usa Hessiana para convergência quadrática. 
        Útil para cálculo de Volatilidade Implícita de alta precisão.
        """
        x = chute.copy()
        historico = []
        
        for k in range(self.max_iter):
            g = gradiente(x)
            h = hessiana(x)
            f_val = funcao_obj(x)
            historico.append(f_val)
            
            if np.linalg.norm(g) < self.tol:
                return ResultadoOtimizacao(True, x, f_val, k, "Convergiu por gradiente", historico)
                
            # Passo de Newton: x_new = x - inv(H)*g
            # Resolve sistema H*d = -g para estabilidade
            try:
                direcao = np.linalg.solve(h, -g)
            except np.linalg.LinAlgError:
                # Fallback para gradiente descendente se H for singular
                direcao = -g * 0.01 
            
            x = x + direcao
            
        return ResultadoOtimizacao(False, x, funcao_obj(x), self.max_iter, "Max iterações atingido", historico)

def teste_otimizacao():
    print("--- Teste de Otimização ---")
    
    # Função Teste: Rosenbrock (Banana function)
    # f(x,y) = (a-x)^2 + b(y-x^2)^2, global min em (a, a^2)
    def rosenbrock(v):
        x, y = v
        return (1 - x)**2 + 100 * (y - x**2)**2
        
    def residuos_rosenbrock(v):
        x, y = v
        return np.array([1 - x, 10 * (y - x**2)]) # Soma quadrados dá a fun acima
    
    opt = OtimizadorFinanceiro()
    
    print("\n1. Teste Levenberg-Marquardt (Trf):")
    res_lm = opt.calibrar_levenberg_marquardt(residuos_rosenbrock, np.array([-1.0, -1.0]))
    print(f"X ótimo: {res_lm.parametros_otimos}, Custo: {res_lm.valor_funcao}")
    
    print("\n2. Teste Simulated Annealing:")
    limites = [(-5.0, 5.0), (-5.0, 5.0)]
    res_sa = opt.simulated_annealing_adaptativo(rosenbrock, limites)
    print(f"X ótimo: {res_sa.parametros_otimos}, Custo: {res_sa.valor_funcao}")

if __name__ == "__main__":
    teste_otimizacao()
