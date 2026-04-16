# Previsão de Receita Previdenciária (RGPS/INSS)

Material suplementar do artigo submetido para revista de economia aplicada.
Compara modelos de previsão para a arrecadação mensal do RGPS/INSS no Brasil,
com validação fora da amostra no período 2020–2024 (incluindo a pandemia de COVID-19).

---

## Modelos Comparados

| Pipeline | Modelos | Framework |
|----------|---------|-----------|
| A — Ensemble | GRU · LSTM · GBRT · Ensemble | TensorFlow / scikit-learn |
| B — TFT | Temporal Fusion Transformer | PyTorch Forecasting |
| C — Baselines | SARIMA · ETS (Holt-Winters) | pmdarima · statsmodels |

Divisão temporal fixa: treino 2004–2015 · validação 2016 · teste 2017–2019 · OOT 2020–2024.

---

## Estrutura do Repositório

```
projeto/
├── data/
│   └── Base_DBP.xlsx          # série histórica mensal (não incluída; ver abaixo)
├── master_pipeline.ipynb      # notebook principal
├── requirements.txt           # dependências com versões fixas
├── run.sh                     # script de inicialização reproduzível
├── README.md                  # este arquivo
└── master_results/            # saídas geradas automaticamente
    ├── ablation_study.csv
    ├── pp_monetary_errors.csv
    ├── pp_subperiod_mape.csv
    ├── comparative_*.xlsx
    └── *.png
```

---

## Requisitos

- **Python** >= 3.10
- Pacotes listados em `requirements.txt`

```bash
pip install -r requirements.txt
```

**Nota sobre `fracdiff`:** o pacote é opcional.
Se não instalado, o notebook usa uma implementação interna equivalente em NumPy.
Para instalar: `pip install fracdiff`

---

## Como Executar

### 1. Preparar os dados

Copie o arquivo `Base_DBP.xlsx` para o diretório `data/`:

```
data/Base_DBP.xlsx
```

### 2. Iniciar o notebook (forma reproduzível)

```bash
bash run.sh
```

O script `run.sh`:
- Define `PYTHONHASHSEED=42` antes de iniciar o Python
- Verifica dependências
- Abre o Jupyter Notebook na interface padrão

Outras opções:

```bash
bash run.sh --lab       # JupyterLab
bash run.sh --execute   # execução headless sem interface (via nbconvert)
```

### 3. Execução manual (menos reproduzível)

```bash
export PYTHONHASHSEED=42
jupyter notebook master_pipeline.ipynb
```

---

## Reprodutibilidade

| Mecanismo | Valor |
|-----------|-------|
| `PYTHONHASHSEED` | 42 (via `run.sh`) |
| `numpy.random.seed` | 42 |
| `tf.random.set_seed` | 42 |
| `pl.seed_everything` | 42 |
| Divisão temporal | Fixa por data (sem aleatoriedade) |
| Versões de pacotes | Fixas em `requirements.txt` |

> **Importante:** `PYTHONHASHSEED` deve ser definido *antes* de iniciar o Python.
> A linha `os.environ["PYTHONHASHSEED"]` dentro do notebook não tem efeito sobre
> o hash do interpretador já em execução. Use `run.sh` para garantia completa.

---

## Saídas Geradas

Todas as saídas são gravadas em `master_results/`:

| Arquivo | Descrição |
|---------|-----------|
| `ablation_study.csv` | 27 combinações (transform × scaler × window) |
| `pp_monetary_errors.csv` | Erros monetários acumulados por modelo (R$ bi) |
| `pp_subperiod_mape.csv` | MAPE% por subperíodo econômico (pré-COVID / choque / recuperação / estabilização) |
| `comparative_*.xlsx` | Tabela comparativa de todos os modelos |
| `execution_times.csv` | Tempo de execução de cada pipeline |
| `*.png` | Gráficos publication-ready |

---

## Citação Sugerida

Se você utilizar este código ou os resultados em publicações acadêmicas, cite:

```
[AUTOR(ES)]. Previsão de Receita Previdenciária com Métodos de Aprendizado de Máquina:
Estudo Comparativo para o RGPS/INSS (2004–2024). [Nome da Revista], [Ano].
DOI: [a preencher após aceite]
```

Código disponível em: [URL do repositório]

---

## Licença

Este material é disponibilizado para fins de revisão e reprodutibilidade científica.
Para outros usos, entre em contato com os autores.

---

*Gerado automaticamente pelo pipeline. Última atualização: ver célula de verificação de integridade no notebook.*
