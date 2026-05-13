# Pipeline de Previsão de Despesas

Notebook Jupyter com pipeline completo de modelagem preditiva para séries temporais mensais de despesas públicas. Compara redes neurais (LSTM, TFT) com modelos clássicos (Ridge, SVR, GBRT, LWR, KNN, AutoARIMA) usando validação temporal rigorosa e otimização automática de hiperparâmetros via Optuna.

---

## Sumário

1. [Estrutura do Notebook](#estrutura-do-notebook)
2. [Dados de Entrada](#dados-de-entrada)
3. [Particionamento Temporal](#particionamento-temporal)
4. [Pré-processamento](#pré-processamento)
5. [Modelos Implementados](#modelos-implementados)
6. [Avaliação](#avaliação)
7. [Saídas Geradas](#saídas-geradas)
8. [Dependências](#dependências)
9. [Como Executar](#como-executar)
10. [Configuração](#configuração)
11. [Resultados de Referência](#resultados-de-referência)

---

## Estrutura do Notebook

O notebook é organizado em **11 células** com separação clara de responsabilidades:

| Célula | Tipo | Conteúdo |
|--------|------|----------|
| 1 | Markdown | Título e descrição do pipeline |
| 2 | Código | **Imports** — todas as bibliotecas em bloco único |
| 3 | Código | **Configuração global** — seeds, splits, hiperparâmetros, diretórios |
| 4 | Markdown | Header das funções utilitárias |
| 5 | Código | **Funções utilitárias** — pré-processamento e métricas compartilhados |
| 6 | Markdown | Header da Etapa 1 |
| 7 | Código | **Etapa 1 — Redes Neurais**: LSTM isolado + TFT |
| 8 | Markdown | Header da Etapa 2 |
| 9 | Código | **Etapa 2 — Modelos Clássicos**: 7 modelos + AutoARIMA |
| 10 | Markdown | Header da Etapa 3 |
| 11 | Código | **Etapa 3 — Visualizações**: 5 gráficos comparativos |

Cada etapa é independente: pode ser re-executada sem rodar as anteriores, desde que os CSVs intermediários já existam (para a Etapa 3).

---

## Dados de Entrada

**Arquivo:** `./data/Base_DBP.csv`

Formato esperado: CSV com duas colunas sem cabeçalho nomeado (colunas renomeadas internamente para `date` e `value`).

```
date,value
2005-01-01,45321.50
2005-02-01,48102.00
...
```

- **Frequência:** mensal (`MS` — Month Start)
- **Unidade:** valor numérico de despesa (ex.: R$)
- **Série histórica usada:** 2005 até 2025 (aproximadamente 269 observações)

---

## Particionamento Temporal

O pipeline usa quatro partições sem embaralhamento (respeita ordem cronológica):

```
|─────────── TRAIN ───────────|─ VAL ─|──────── TEST ────────|──────────── HOLDOUT ────────────|
         até 2015-12           2016     2017-01 a 2019-12         2020-01 em diante
         (~149 obs)           (~12 obs)    (~36 obs)                  (~72 obs)
```

| Partição | Período | Uso |
|----------|---------|-----|
| **Train** | até `2015-12` | Ajuste dos modelos |
| **Val** | `2016` (12 meses) | Otimização de hiperparâmetros (Optuna) |
| **Test** | `2017-01` a `2019-12` (36 meses) | Avaliação walk-forward 1-step-ahead |
| **Holdout** | `2020-01` em diante (~72 meses) | Avaliação de previsão recursiva multi-step |

O **Holdout** simula o uso real do modelo: prever múltiplos meses no futuro sem acesso a valores observados entre as previsões.

---

## Pré-processamento

Todas as etapas aplicam o mesmo pipeline de transformação, implementado nas funções utilitárias (Célula 5):

### 1. Diferenciação Sazonal (Lag-12)

```
z_t = y_t - y_{t-12}
```

Remove a sazonalidade anual da série antes de modelar. Os primeiros 12 valores ficam como `NaN` e são descartados.

**Motivação:** a série de despesas públicas tem forte componente sazonal mensal (padrão que se repete a cada 12 meses). A diferenciação remove essa estrutura, deixando o modelo focar nas variações residuais.

### 2. Normalização (MinMaxScaler)

Aplica `MinMaxScaler(feature_range=(-1, 1))` **ajustado apenas no Train**, evitando vazamento de informação do futuro.

### 3. Janela Deslizante (Supervised Learning)

Transforma a série em pares `(X, y)` com janela de `WINDOW_SIZE = 12` passos:

```
X = [z_{t-12}, z_{t-11}, ..., z_{t-1}]   →   y = z_t
```

Para modelos neurais (LSTM), o array X é reforçado para shape `(samples, timesteps, 1)`.

---

## Modelos Implementados

### Etapa 1 — Redes Neurais

#### LSTM Isolado (Keras/TensorFlow)

Arquitetura Sequential:
```
Input(shape=(12, 1))
→ LSTM(units)
→ Dropout(dropout)
→ Dense(16, activation='relu')
→ Dense(1)
```

Otimização de hiperparâmetros via **Optuna** (TPE Sampler, `N_TRIALS_LSTM = 25` trials):
- `units`: {16, 32, ..., 128}
- `dropout`: [0.0, 0.4]
- `lr`: [1e-4, 1e-2] (log)
- `batch_size`: {8, 16, 32}

Treinamento com **EarlyStopping** (`patience=15` na busca, `patience=20` no modelo final).

---

#### TFT — Temporal Fusion Transformer (darts)

Modelo baseado em atenção multi-head para séries temporais. Configuração univariada com `add_relative_index=True` como única feature derivada do timestamp.

Otimização via Optuna (`N_TRIALS_TFT = 12` trials, mais lento):
- `hidden_size`: {8, 16, 32}
- `lstm_layers`: {1, 2}
- `num_heads`: {1, 2, 4}
- `dropout`: [0.0, 0.3]
- `lr`: [1e-4, 1e-2] (log)

**Nota:** o TFT é opcional. Se `darts` não estiver instalado, é ignorado com aviso.

---

### Etapa 2 — Modelos Clássicos

Todos os modelos recebem como entrada a janela deslizante `X ∈ R^12` e são otimizados via Optuna (`N_TRIALS = 50`).

| Modelo | Classe | Hiperparâmetros otimizados |
|--------|--------|---------------------------|
| **Ridge** | `sklearn.linear_model.Ridge` | `alpha` ∈ [1e-4, 1e3] |
| **SVR** | `sklearn.svm.SVR` (kernel RBF) | `C`, `gamma`, `epsilon` |
| **GBRT** | `GradientBoostingRegressor` | `n_estimators`, `max_depth`, `lr`, `subsample`, `min_samples_leaf` |
| **LWR** | `LocallyWeightedRegression` (custom) | `bandwidth` ∈ [0.05, 5.0] |
| **Nearest (1-NN)** | `KNeighborsRegressor(k=1)` | — (sem otimização) |
| **KNN** | `KNeighborsRegressor` | `n_neighbors` ∈ [2, 20] |
| **AutoARIMA** | `pmdarima.auto_arima` | busca automática SARIMA (p,d,q)(P,D,Q,12) |

#### LWR — Locally Weighted Regression (implementação própria)

Baseado em Atkeson (1997). Para cada query `x*`, calcula pesos gaussianos:

```
w_i = exp(-||x_i - x*||² / (2h²))
```

Resolve uma regressão linear ponderada local com regularização Ridge para cada ponto de predição. Custo O(N·d) por query — adequado para a escala da série.

#### AutoARIMA

Usa `pmdarima.auto_arima` com busca stepwise por `(p,d,q)(P,D,Q,12)`. Resultado típico: `(0,1,1)×(1,0,2,12)`. **Opcional** — ignorado se `pmdarima` não estiver instalado.

O treino final de todos os modelos clássicos usa **train + val** combinados (mais dados com hiperparâmetros já fixados).

---

## Avaliação

### Walk-forward (Test — 2017 a 2019)

Para cada mês `t` do Test:
1. Usa a janela `z_{t-12:t}` com valores **reais** até `t-1`
2. Prediz `ŷ_t`
3. Avança para `t+1` (sem atualizar o modelo)

Simula o cenário operacional mês a mês com acesso ao valor real anterior.

### Recursivo (Holdout — 2020 em diante)

Para cada mês `t` do Holdout:
1. A previsão `ŷ_t` entra no histórico como se fosse valor real
2. A próxima janela já usa previsões anteriores
3. Os erros se acumulam ao longo do horizonte

Mede a capacidade do modelo de sustentar previsões de longo prazo.

### Métricas reportadas

| Métrica | Fórmula | Interpretação |
|---------|---------|---------------|
| **MAE** | `mean(|y - ŷ|)` | Erro absoluto médio (unidade original) |
| **RMSE** | `sqrt(mean((y - ŷ)²))` | Penaliza erros grandes |
| **MAPE (%)** | `mean(|y - ŷ| / y) × 100` | Erro percentual médio |
| **Bias (%)** | `mean((y - ŷ) / y) × 100` | Direção sistemática do erro (+ = subestimação) |

As métricas são calculadas para **Test** e **Holdout** separadamente.

---

## Saídas Geradas

### CSVs (`./outputs/`)

| Arquivo | Conteúdo |
|---------|----------|
| `neurais_metrics.csv` | MAE, RMSE, MAPE, Bias de LSTM e TFT nos dois regimes |
| `neurais_previsoes.csv` | Série real + previsões de LSTM e TFT (Test + Holdout) |
| `classicos_metrics.csv` | Métricas dos 7 modelos clássicos |
| `classicos_previsoes.csv` | Série real + previsões dos 7 modelos |

### Gráficos (`./outputs/`)

| Arquivo | Conteúdo |
|---------|----------|
| `neurais_comparativo.png` | Série completa + LSTM vs TFT (Test e Holdout) |
| `classicos_comparativo.png` | Grade com 7 subplots — cada modelo vs realizado |

### Gráficos de Análise (`./outputs/charts/`)

| Arquivo | Conteúdo |
|---------|----------|
| `01_holdout_redes_neurais.png` | Zoom no Holdout com LSTM e TFT, ordenados por MAPE |
| `02_holdout_classicos.png` | Zoom no Holdout com os 7 modelos clássicos |
| `03_holdout_todos.png` | Zoom no Holdout com todos os modelos juntos |
| `04_metricas_comparativo.png` | Dashboard 2×2: MAPE Test vs Holdout · Bias · Heatmap por ano · Boxplot de erros |
| `05_analise_serie.png` | STL (tendência + sazonalidade + resíduo) · Boxplot mensal · Crescimento YoY |

---

## Dependências

### Obrigatórias

```
numpy
pandas
matplotlib
scikit-learn
optuna
tensorflow >= 2.10
```

### Opcionais

| Biblioteca | Uso | Instalação |
|-----------|-----|-----------|
| `darts` | Modelo TFT | `pip install darts` |
| `pmdarima` | AutoARIMA | `pip install pmdarima` |
| `statsmodels` | Decomposição STL nos gráficos | `pip install statsmodels` |

Se uma dependência opcional não estiver instalada, o modelo/gráfico correspondente é ignorado com aviso, e o restante do pipeline continua normalmente.

### Instalação completa

```bash
pip install numpy pandas matplotlib scikit-learn optuna tensorflow
pip install darts pmdarima statsmodels
```

---

## Como Executar

Execute as células em ordem de cima para baixo no Jupyter:

```
Célula 2 → Célula 3 → Célula 5 → Célula 7 → Célula 9 → Célula 11
(imports)  (config)  (utils)    (neural)   (classical) (viz)
```

As etapas 1 e 2 são independentes entre si e podem ser executadas em qualquer ordem. A Etapa 3 (visualizações) requer que os CSVs das etapas anteriores já existam em `./outputs/`.

### Tempo estimado de execução (CPU)

| Etapa | Tempo aproximado |
|-------|-----------------|
| Etapa 1 — LSTM (25 trials Optuna) | ~5–10 min |
| Etapa 1 — TFT (12 trials Optuna) | ~15–25 min |
| Etapa 2 — Modelos clássicos (50 trials cada) | ~3–8 min |
| Etapa 3 — Visualizações | < 1 min |

---

## Configuração

Todos os parâmetros estão centralizados na **Célula 3**:

```python
# Splits temporais
DATA_PATH = "./data/Base_DBP.csv"
TRAIN_END = "2015-12"
VAL_END   = "2016-12"
TEST_END  = "2019-12"

# Janela e sazonalidade
SEASONAL_LAG = 12    # diferenciação lag-12
WINDOW_SIZE  = 12    # janela de input dos modelos

# Trials do Optuna
N_TRIALS       = 50   # modelos clássicos
N_TRIALS_LSTM  = 25
N_TRIALS_TFT   = 12

# Treinamento neural
EPOCHS_MAX_LSTM = 200
EPOCHS_MAX_TFT  = 80
BATCH_SIZE      = 16

# Reprodutibilidade
SEED = 42
```

Para reduzir o tempo de execução, diminua `N_TRIALS_*`. Para aumentar a qualidade da busca, aumente-os.

---

## Resultados de Referência

Resultados obtidos com `SEED=42`, `WINDOW_SIZE=12`, `N_TRIALS_LSTM=25`, `N_TRIALS_TFT=12`, `N_TRIALS=50`:

### Test (2017–2019) — Walk-forward 1-step-ahead

| Modelo | MAE | MAPE (%) |
|--------|-----|----------|
| KNN | ~1.755 | ~3.52% |
| SVR | ~2.018 | ~4.17% |
| Nearest (1-NN) | ~2.106 | ~4.24% |
| AutoARIMA | ~2.238 | ~4.47% |
| LSTM isolado | ~2.306 | ~4.87% |

### Holdout (>2019) — Recursivo multi-step

| Modelo | MAPE (%) |
|--------|----------|
| Ridge | ~13.70% |
| Lazy (LWR) | ~13.77% |
| AutoARIMA | ~13.89% |
| Nearest (1-NN) | ~14.81% |
| LSTM isolado | ~16.07% |

> O degradação entre Test e Holdout é esperada: no Holdout os erros se acumulam recursivamente, especialmente a partir de 2020 onde choques externos (pandemia, inflação) não fazem parte do padrão histórico aprendido.
