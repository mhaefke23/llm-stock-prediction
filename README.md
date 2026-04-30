# LLM-Augmented Stock Return Prediction

CBS MSc AI & Machine Learning course project — Copenhagen Business School

## Research question

> Does providing an LLM with financial news improve next-day stock return direction prediction compared to price data alone — and how do LLM-based approaches compare to a traditional LSTM baseline?

## Experimental conditions

| Condition | Description | Inputs | Method |
|-----------|-------------|--------|--------|
| **B**  | LSTM baseline  | OHLCV + technical indicators | 2-layer LSTM → log return regression |
| **L1** | LLM price-only | Last 10 days of price data   | GPT-5-nano → up / down + confidence |
| **L2** | LLM news-only  | Top-5 Massive article summaries | GPT-5-nano → up / down + confidence |
| **L3** | LLM price+news | Price data + news summaries  | GPT-5-nano → up / down + confidence |

All four conditions are evaluated on **the same 188 trading days** (Jul 2025 – Apr 2026) using a walk-forward rolling-window setup for fair comparison.

## Data sources

| Source | Contents | File |
|--------|----------|------|
| [yfinance](https://pypi.org/project/yfinance/) | AAPL OHLCV, Jul 2024 – Apr 2026 | `data/processed/prices.csv` |
| [Massive / Polygon](https://polygon.io/) | News article summaries + AAPL sentiment | `data/processed/news_cache.csv` |

## Evaluation setup

- **Warm-up:** first 250 trading days (LSTM training only, no evaluation)
- **Evaluation period:** ~188 trading days
- **LSTM rolling window:** 250 days, retrained every 20 evaluation days
- **LLM:** no training; queried fresh for each evaluation day
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

# 3. LSTM baseline (condition B)
python src/train_eval.py

# 4. LLM conditions (dry-run first, then full)
python src/llm_predict.py --condition L1 --dry-run
python src/llm_predict.py --condition L1
python src/llm_predict.py --condition L2
python src/llm_predict.py --condition L3

# 5. Analysis notebook
jupyter notebook notebooks/analysis.ipynb
```

## Project structure

```
├── src/
│   ├── fetch_data.py      # Price download + feature engineering
│   ├── fetch_news.py      # Massive news bulk fetch + cache
│   ├── model.py           # LSTMForecaster (2-layer, hidden=64)
│   ├── train_eval.py      # Walk-forward evaluation for condition B
│   └── llm_predict.py     # LLM conditions L1 / L2 / L3
├── notebooks/
│   └── analysis.ipynb     # Charts and metrics for all conditions
├── data/processed/        # prices.csv, news_cache.csv
├── results/               # predictions_B/L1/L2/L3.csv, figures/
├── requirements.txt
└── .env                   # API keys (gitignored)
```

## Requirements

See `requirements.txt`. Key dependencies: `yfinance`, `torch`, `openai`,
`polygon-api-client`, `scikit-learn`, `pandas`, `matplotlib`.
