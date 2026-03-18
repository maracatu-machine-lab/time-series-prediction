# Time Series Prediction — DBP Forecasting

A unified machine learning pipeline for forecasting **Diastolic Blood Pressure (DBP)** time series, combining classical ensemble models with a Temporal Fusion Transformer.

---

## Overview

The project is organized around two pipelines that can be run independently or together via a single master notebook:

| Pipeline | Models | Framework |
|---|---|---|
| **A — Ensemble Walk-Forward** | GRU · LSTM · GBRT | TensorFlow / Keras + scikit-learn |
| **B — TFT v5** | Temporal Fusion Transformer | PyTorch Forecasting + Optuna |

Results from both pipelines are compared side-by-side in the final section of `master_pipeline.ipynb`.

---

## Notebooks

| File | Description |
|---|---|
| `master_pipeline.ipynb` | Main entry point. Runs Pipeline A and/or B, generates all plots and exports results. |
| `build_master_notebook.py` | Script that programmatically generates `master_pipeline.ipynb` from source. Run with `python build_master_notebook.py`. |
| `part-fase7.ipynb` | Experimental notebook for phase 7 of the research. |
| `tft1-Copy1.ipynb` | Standalone TFT v5 development notebook with interpretability plots. |

---

## Pipeline A — Ensemble Walk-Forward (GRU + LSTM + GBRT)

- **Walk-forward validation** with configurable refit interval
- Three time-series transformations applied independently:
  - `sazonal_lag12` — Seasonal lag-12 decomposition
  - `log_retorno` — Log return
  - `retorno_pct` — Percentage return
- Each transformation trains GRU, LSTM, and GBRT separately; predictions are **ensembled** (averaged)
- A **30-step future forecast** is generated using the best-performing transformation (`sazonal_lag12`)
- Outputs saved to `master_results/`:
  - `pA_*.csv` — fold-level predictions per transformation
  - `comparative_MinMaxScaler.xlsx` — aggregated metrics table
  - Walk-forward, R², and forecast PNGs

## Pipeline B — TFT v5 (Temporal Fusion Transformer)

- **Expanding-window cross-validation** (3 folds)
- **Optuna hyperparameter optimization** (20 trials per fold)
- Rich feature set: moving averages (3/6/12m), z-score, volatility, seasonal index, calendar encodings
- Interpretability outputs: variable importance, attention weights, confidence intervals
- Disabled by default — requires `pytorch-forecasting`. See [Enabling TFT](#enabling-tft) below.

---

## Metrics

All models are evaluated with:

| Metric | Description |
|---|---|
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| MAPE % | Mean Absolute Percentage Error |
| R² | Coefficient of Determination |
| Viés % | Bias percentage |
| Estab % | Stability percentage |

---

## Configuration

Everything is controlled from a **single cell** at the top of `master_pipeline.ipynb`:

```python
SCALER_TYPE      = "MinMaxScaler"       # "MinMaxScaler" | "StandardScaler" | "RobustScaler"
DATA_PATH        = r'...\Base_DBP.xlsx'
PIPELINES_TO_RUN = ["ensemble_gru_lstm_gbrt"]  # add "tft" to enable Pipeline B
TRANSFORMATIONS  = ["sazonal_lag12", "log_retorno", "retorno_pct"]
WINDOW_SIZE      = 14
```

Model architectures (GRU, LSTM, GBRT, TFT) are also configurable in the same cell.

---

## Enabling TFT

1. Uncomment `"tft"` in `PIPELINES_TO_RUN`:
   ```python
   PIPELINES_TO_RUN = [
       "ensemble_gru_lstm_gbrt",
       "tft",   # ← uncomment this
   ]
   ```
2. Install the extra dependency:
   ```bash
   pip install pytorch-forecasting
   ```
3. Run all cells (`Kernel → Restart & Run All`).

---

## Requirements

```bash
pip install tensorflow keras scikit-learn pandas numpy openpyxl nbformat
# For Pipeline B (TFT):
pip install pytorch-forecasting optuna
```

Python 3.10+ recommended.

---

## Results

Current results (session `2026-03-09`, `MinMaxScaler`) are stored in `master_results/`:

- `comparative_MinMaxScaler.xlsx` — full metrics table for all Pipeline A models
- `comparative_r2_MinMaxScaler.png` — R² bar chart comparison
- `comparativo_MinMaxScaler_sazonal.png` — seasonal comparison plot
- `pA_forecast30y_MinMaxScaler.csv` — 30-step future forecast values
