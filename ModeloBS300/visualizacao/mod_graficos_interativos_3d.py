"""
Módulo de Visualização Científica e Gráficos Interativos 3D
===========================================================

Responsável pela renderização de superfícies de volatilidade, cones de probabilidade
e comparação visual de cenários de previsão (Real vs Previsto).
Utiliza Plotly para interatividade web/notebook e Matplotlib/Seaborn para relatórios estáticos de alta resolução (fundo de tese/paper).

Autor: Luiz Tiago Wilcke
Data: 2026-01-19
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
import logging

logger = logging.getLogger("VisualizacaoAvancada")

# Estilo global Matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("paper", font_scale=1.2)

class VisualizadorFinanceiro:
    """Fabrica de gráficos financeiros complexos."""

    @staticmethod
    def plotar_superficie_volatilidade(
        strikes: np.ndarray,
        vencimentos: np.ndarray,
        volatilidades: np.ndarray,
        titulo: str = "Superfície de Volatilidade Implícita (Wilcke Smirk)"
    ):
        """
        Plota superfície 3D (Strike x Tempo x Vol).
        Entrada deve ser meshgrid.
        """
        fig = go.Figure(data=[go.Surface(
            z=volatilidades,
            x=strikes,
            y=vencimentos,
            colorscale='Viridis',
            opacity=0.9
        )])
        
        fig.update_layout(
            title=titulo,
            scene=dict(
                xaxis_title='Strike (K)',
                yaxis_title='Tempo até Vencimento (T)',
                zaxis_title='Volatilidade Implícita ($\sigma$)'
            ),
            width=900, height=600,
            autosize=False
        )
        return fig

    @staticmethod
    def plotar_previsao_fan_chart(
        historico: pd.Series,
        previsao_media: pd.Series,
        intervalos_confianca: List[Tuple[pd.Series, pd.Series]], # [(low95, high95), (low50, high50)]
        titulo: str = "Previsão de Preços 2026-2028 (Fan Chart)"
    ):
        """
        Gera gráfico de leque (Bank of England style) mostrando incerteza crescente.
        Ideal para previsões de longo prazo.
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plot Histórico
        ax.plot(historico.index, historico.values, color='black', label='Histórico Real', linewidth=1.5)
        
        # Plot Previsão Média
        ax.plot(previsao_media.index, previsao_media.values, color='navy', linestyle='--', label='Previsão Média (Modelo Híbrido)')
        
        # Plot Intervalos (Fan)
        alphas = [0.1, 0.3] # Transparência
        cores = ['lightblue', 'blue']
        
        for i, (lower, upper) in enumerate(intervalos_confianca):
            ax.fill_between(
                previsao_media.index, 
                lower, upper, 
                color=cores[i], 
                alpha=alphas[i],
                label=f'Intervalo Confiança {95 if i==0 else 50}%'
            )
            
        ax.set_title(titulo, fontsize=16, fontweight='bold')
        ax.set_xlabel("Ano", fontsize=12)
        ax.set_ylabel("Preço do Ativo (BRL)", fontsize=12)
        ax.legend(loc='upper left')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Formatar eixo X datas
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        return fig

    @staticmethod
    def plotar_comparativo_multi_modelos(
        dados_reais: pd.Series,
        preds_dict: Dict[str, pd.Series], # {'LSTM': s1, 'ARIMA': s2, 'Wilcke': s3}
        titulo: str = "Batalha de Modelos: Desempenho Preditivo"
    ):
        """Compara visualmente múltiplas curvas de previsão."""
        fig = go.Figure()
        
        # Real
        fig.add_trace(go.Scatter(
            x=dados_reais.index, y=dados_reais.values,
            mode='lines', name='Realizado',
            line=dict(color='black', width=3)
        ))
        
        cores = ['red', 'green', 'blue', 'orange', 'purple']
        
        for i, (nome_modelo, serie_pred) in enumerate(preds_dict.items()):
            fig.add_trace(go.Scatter(
                x=serie_pred.index, y=serie_pred.values,
                mode='lines', name=f'Previsto ({nome_modelo})',
                line=dict(color=cores[i % len(cores)], width=2, dash='dot')
            ))
            
        fig.update_layout(
            title=titulo,
            xaxis_title="Data",
            yaxis_title="Valor",
            hovermode="x unified",
            template="plotly_white"
        )
        return fig

    @staticmethod
    def plotar_simulacao_monte_carlo(
        caminhos: np.ndarray, 
        titulo: str = "Simulação Monte Carlo (5000 Caminhos)"
    ):
        """Visualização leve de milhares de caminhos (Downsampling se necessário)."""
        passos, n_sims = caminhos.shape
        # Se muitos, plotar apenas amostra
        n_plot = min(n_sims, 200)
        
        plt.figure(figsize=(12, 6))
        plt.plot(caminhos[:, :n_plot], color='blue', alpha=0.05, linewidth=0.5)
        plt.plot(np.mean(caminhos, axis=1), color='red', linewidth=2, label='Média Esperada')
        
        plt.title(titulo)
        plt.ylabel("Preço")
        plt.xlabel("Passos de Tempo")
        plt.legend()
        return plt.gcf()

def teste_viz():
    print("--- Teste de Visualização ---")
    
    viz = VisualizadorFinanceiro()
    
    # Fake Data
    datas_hist = pd.date_range("2025-01-01", periods=100)
    hist = pd.Series(np.linspace(100, 120, 100), index=datas_hist)
    
    datas_fut = pd.date_range("2026-01-01", periods=50)
    pred_media = pd.Series(np.linspace(120, 140, 50), index=datas_fut)
    
    low95 = pred_media * 0.90
    high95 = pred_media * 1.10
    
    # Teste Matplotlib Fan Chart
    fig1 = viz.plotar_previsao_fan_chart(hist, pred_media, [(low95, high95)])
    fig1.savefig("teste_fan_chart.png")
    print("Fan Chart salvo como teste_fan_chart.png")

if __name__ == "__main__":
    teste_viz()
