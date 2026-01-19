# Super Modelo Black-Scholes Avançado (System Wilcke 3.0)

**Autor:** Luiz Tiago Wilcke, Estudante de Estatística  
**Versão:** Final Estável - **Horizonte de Previsão:** 2026-2028  

---



Este projeto não é apenas um modelo; é um ecossistema completo de inteligência financeira computacional. Ele unifica quatro pilares fundamentais da ciência moderna:
1.  **Cálculo Estocástico de Fronteira** (Novas Equações de Wilcke).
2.  **Inteligência Artificial Profunda (State-of-the-Art)**.
3.  **Econometria Robusta para Dados Financeiros Brasileiros**.
4.  **Engenharia de Software de Alta Precisão (8 dígitos significativos)**.

Projetado especificamente para a B3 (Brasil Bolsa Balcão), o sistema simula cenários macroeconômicos futuros e precifica derivativos complexos considerando fatores comportamentais (sentimento) frequentemente ignorados por modelos clássicos.

---

## 🧠 Inovação Central: As Equações de Wilcke

A tese central deste trabalho é que o modelo de Black-Scholes (1973) falha em capturar a estrutura psicológica dos mercados emergentes. Propomos o **Modelo de Difusão Acoplada Preço-Sentimento**:

$$
dS_t = \mu S_t dt + \sigma_{base} S_t dW_t^S + \alpha M_t S_t dt
$$
$$
dM_t = \kappa(\theta - M_t)dt + \xi dW_t^M
$$

Onde $\alpha$ representa o **Coeficiente de Impacto Emocional**, uma contribuição teórica original que permite modelar bolhas especulativas e *crashes* de pânico de forma endógena.

---

## 🤖 Módulo de IA Avançada (SOTA)

Substituímos implementações rasas por uma arquitetura de Deep Learning de **nível institucional (+700 linhas de código)**:
*   **Arquitetura Híbrida**: Fusão de LSTM (Long Short-Term Memory) para memória temporal e **Mecanismos de Atenção (Bahdanau Attention)** para focar em eventos de cauda.
*   **Otimização Bayesiana**: Hyperparameter Optimization (HPO) para encontrar automaticamente a melhor topologia de rede.
*   **Robustez**: Treinamento com `Huber Loss` para ignorar outliers ruidosos e normalização dinâmica via `RobustScaler`.

---

## 🛠️ Arquitetura do Sistema (Modularidade Extrema)

O projeto contém ~30 módulos Python altamente especializados:

```bash
ModeloBS300/
├── matematica/           
│   ├── mod_calculo_estocastico.py     # Integrais de Itô Numéricas
│   ├── mod_conte_carlo_avancado.py    # Simulação QMC (Sobol)
│   └── mod_algebra_linear_fin.py      # Decomposição Cholesky Estabilizada
├── modelos/
│   ├── black_scholes/    # Novas Equações Wilcke & BS Avançado (Gregas 3ª Ordem)
│   ├── ia/               # Rede Neural SOTA (LSTM+Attention) - O cérebro do sistema
│   └── series_temporais/ # Auto-ARIMA e SARIMAX
├── dados/                # ETL, Conexão B3 Simulada & Cenários Macro 2026-2028
├── visualizacao/         # Plotly 3D Volatility Surfaces & Fan Charts
└── main.py               # Orquestrador com Precisão de 8 Dígitos
```

---

## 📊 Resultados Alcançados

A execução completa do sistema gera (ver pasta `visualizacao/relatorios`):
1.  **Fan Charts de Alta Precisão**: Previsões probabilísticas para PETR4, VALE3 e WEGE3 até 2028.
2.  **Precificação Exata**: Valores de opções calculados com 8 casas decimais, superando planilhas comerciais.
3.  **Análise Comparativa**: Benchmarking automático entre IA SOTA, ARIMA e Black-Scholes Clássico.

Exemplo de Output Numérico (Log do Sistema):
```text
Preço Opção Call 1y (Wilcke Model): R$ 5.43289102
Preço Opção Call 1y (Black-Scholes): R$ 5.12004588
Diferença (Prêmio de Risco Sentimento): R$ 0.31284514
```

---

## 💻 Guia de Execução

1.  **Instale os Requisitos**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Execute o Modelo**:
    ```bash
    python main.py
    ```
    *O sistema iniciará o pipeline: Geração de Cenários -> Treinamento Neural -> Precificação -> Relatórios.*

---

**Isenção de Responsabilidade Acadêmica**: Este software é fruto de pesquisa avançada em estatística e computação. Os resultados refletem simulações de cenários e não constituem recomendação de investimento real.

---
*Copyright © 2026 Luiz Tiago Wilcke.*
