# LLM-Augmented Stock Return Prediction

CBS MSc AI & Machine Learning course project — Copenhagen Business School

## Research question

> How can large language models be used for stock return prediction, and how does the availability of news data affect predictive performance across LLM-based and trained machine learning approaches?

## Experimental conditions

| Condition | Description | Inputs | Method |
|-----------|-------------|--------|--------|
| **B**  | LSTM regressor | OHLCV + technical indicators | 2-layer LSTM → log return → direction |
| **B1** | LSTM classifier (single-day) | OHLCV + technicals, seq_len=1 | LSTM with BCELoss → direction |
| **B2** | LSTM classifier (sequence) | OHLCV + technicals, seq_len=20 | LSTM with BCELoss → direction |
| **L1** | LLM — price only | Last 10 days of price data | GPT-5-nano → up / down + confidence |
| **L2** | LLM — news only | Top-5 Massive article summaries | GPT-5-nano → up / down + confidence |
| **L3** | LLM — price + news | Price data + news summaries | GPT-5-nano → up / down + confidence |
| **X1** | XGB regressor (single-day) | Current-day OHLCV features | XGBoost → log return → direction |
| **X2** | XGB regressor (20-day lags) | Flattened 20-day features | XGBoost → log return → direction |
| **X3** | XGB classifier (single-day) | Current-day OHLCV features | XGBoost → direction |
| **X4** | XGB classifier (20-day lags) | Flattened 20-day features | XGBoost → direction |
| **XN** | XGB classifier + news | Current-day features + PCA news embeddings | XGBoost → direction |

All 11 conditions are evaluated on **the same 188 trading days** (Jul 2025 – Apr 2026) using a walk-forward rolling-window setup.

## Results

| Condition | Dir. Accuracy | Pred. "up" rate |
|-----------|:---:|:---:|
| B — LSTM regressor | 52.7% | 47.9% |
| B1 — LSTM clf (single-day) | 52.1% | 42.0% |
| B2 — LSTM clf (sequence) | 50.0% | 51.6% |
| L1 — LLM price only | 46.8% | 63.3% |
| L2 — LLM news only | 48.4% | 68.1% |
| L3 — LLM price+news | 52.7% | 64.9% |
| X1 — XGB regressor (single-day) | 45.2% | 47.9% |
| X2 — XGB regressor (20-day lags) | 46.8% | 53.7% |
| **X3 — XGB clf (single-day)** | **56.4%** | 72.9% |
| X4 — XGB clf (20-day lags) | 53.7% | 87.2% |
| XN — XGB clf + news | 52.7% | 84.0% |

Random baseline: **50.0%**. Actual market up-rate: **51.6%**. X3's lead is partly attributable to upward prediction bias — see the Discussion in `notebooks/analysis.ipynb`.

## Data sources

| Source | Contents | File |
|--------|----------|------|
| [yfinance](https://pypi.org/project/yfinance/) | AAPL OHLCV, Jul 2024 – Apr 2026 | `data/processed/prices.csv` |
| [Massive / Polygon](https://polygon.io/) | News article summaries + AAPL sentiment | `data/processed/news_cache.csv` |
| OpenAI GPT-5-nano | Structured daily news reports | `data/processed/news_reports.csv` |

## Evaluation setup

- **Warm-up:** first 250 trading days (model training only, no evaluation)
- **Evaluation period:** 188 trading days
- **Rolling window:** 250 days, models retrained every 20 evaluation days
- **LLM conditions:** no training; queried fresh for each evaluation day
- **Primary metric:** directional accuracy (50% random baseline)

## Setup

```bash
git clone <repo-url>
cd llm-stock-prediction
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` in the project root:
```
OPENAI_API_KEY=sk-...
MASSIVE_API_KEY=...
```

## Running the pipeline

```bash
# 1. Download price data
python src/fetch_data.py

# 2. Fetch news (single bulk API call)
python src/fetch_news.py

# 3. Generate structured reports + embeddings (needed for XN)
python src/generate_reports.py
python src/embed_reports.py

# 4. LSTM conditions
python src/train_eval.py                        # condition B
python src/train_eval_lstm_clf.py --condition B1
python src/train_eval_lstm_clf.py --condition B2

# 5. XGBoost conditions
python src/train_eval_xgb.py --condition X1    # repeat for X2, X3, X4
python src/train_eval_xgb_news.py              # condition XN

# 6. LLM conditions
python src/llm_predict.py --condition L1       # repeat for L2, L3

# 7. Analysis notebook
jupyter notebook notebooks/analysis.ipynb
# Export PDF (not committed to repo — requires XeLaTeX)
# jupyter nbconvert --to pdf --no-input notebooks/analysis.ipynb
```

## Project structure

```
├── src/
│   ├── fetch_data.py           # Price download + feature engineering
│   ├── fetch_news.py           # Massive news bulk fetch + cache
│   ├── generate_reports.py     # GPT-5-nano structured daily report generation
│   ├── embed_reports.py        # Sentence-transformer embedding of reports
│   ├── model.py                # LSTMForecaster (2-layer, hidden=64)
│   ├── train_eval.py           # Walk-forward evaluation — condition B
│   ├── train_eval_lstm_clf.py  # Walk-forward evaluation — conditions B1, B2
│   ├── train_eval_xgb.py       # Walk-forward evaluation — conditions X1–X4
│   ├── train_eval_xgb_news.py  # Walk-forward evaluation — condition XN
│   └── llm_predict.py          # LLM conditions L1 / L2 / L3
├── notebooks/
│   ├── analysis.ipynb              # Charts, metrics, and discussion (pre-executed)
│   └── news_api_coverage_test.ipynb  # News API coverage exploration
├── data/processed/             # prices.csv, news_cache.csv, news_reports.csv
├── results/                    # predictions_{B,B1,B2,L1,L2,L3,X1,X2,X3,X4,XN}.csv, figures/
├── requirements.txt
└── .env                        # API keys (gitignored)
```

## Requirements

See `requirements.txt`. Key dependencies: `yfinance`, `torch`, `xgboost`, `openai`,
`sentence-transformers`, `scikit-learn`, `pandas`, `matplotlib`.
