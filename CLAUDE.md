# CLAUDE.md — LLM-Augmented Stock Return Prediction

## Project overview

CBS MSc AI & Machine Learning course project comparing four conditions for predicting
next-day AAPL stock return direction over a walk-forward rolling-window evaluation
(Jul 2025 – Apr 2026, 188 evaluation days).

| Condition | Name | Method |
|-----------|------|--------|
| **B**  | LSTM baseline  | 2-layer LSTM on OHLCV + technicals → log return → direction |
| **L1** | LLM price-only | GPT-5-nano on last 10 days of price data → up/down |
| **L2** | LLM news-only  | GPT-5-nano on top-5 Massive article summaries → up/down |
| **L3** | LLM price+news | GPT-5-nano on price + news → up/down |

## Repository layout

```
llm-stock-prediction/
├── data/
│   ├── raw/                        # (empty — no raw files committed)
│   └── processed/
│       ├── prices.csv              # AAPL OHLCV + features, Jul 2024 – Apr 2026
│       └── news_cache.csv          # Massive/Polygon news, one row per trading day
├── results/
│   ├── predictions_B.csv           # LSTM walk-forward predictions
│   ├── predictions_L1.csv          # LLM price-only predictions
│   ├── predictions_L2.csv          # LLM news-only predictions
│   ├── predictions_L3.csv          # LLM price+news predictions
│   └── figures/                    # Charts saved by the notebook
├── src/
│   ├── fetch_data.py               # yfinance download + feature engineering
│   ├── fetch_news.py               # Massive/Polygon bulk news fetch + cache
│   ├── model.py                    # LSTMForecaster class definition
│   ├── train_eval.py               # Walk-forward evaluation for condition B
│   └── llm_predict.py              # LLM conditions L1 / L2 / L3
├── notebooks/
│   └── analysis.ipynb              # All charts and metrics
├── requirements.txt
├── .env                            # API keys (gitignored)
└── .gitignore
```

## Environment setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file with:
```
OPENAI_API_KEY=sk-...
MASSIVE_API_KEY=...
```

## Running the pipeline

Run scripts **in order** from the project root:

### Step 1 — Fetch price data
```bash
python src/fetch_data.py
```
Downloads AAPL OHLCV from yfinance, computes features, saves `data/processed/prices.csv`.

### Step 2 — Fetch news
```bash
python src/fetch_news.py          # bulk fetch (~3500 articles in one call)
python src/fetch_news.py --force  # re-fetch and overwrite cache
```
Saves `data/processed/news_cache.csv` with text, article count, no_news flag, and
AAPL-specific sentiment from Massive insights.

### Step 3 — Train and evaluate condition B (LSTM)
```bash
python src/train_eval.py --eval-days 20   # quick test (first 20 eval days)
python src/train_eval.py                  # full run (~188 eval days)
```

### Step 4 — Evaluate LLM conditions
```bash
# Dry-run first to verify prompts
python src/llm_predict.py --condition L1 --dry-run
python src/llm_predict.py --condition L2 --dry-run
python src/llm_predict.py --condition L3 --dry-run

# Small live test
python src/llm_predict.py --condition L1 --days 5

# Full runs (resume-safe — interrupted runs continue from cache)
python src/llm_predict.py --condition L1
python src/llm_predict.py --condition L2
python src/llm_predict.py --condition L3
```

### Step 5 — Analysis notebook
```bash
jupyter notebook notebooks/analysis.ipynb
```
Run all cells top-to-bottom. Figures are saved to `results/figures/`.

## Key design decisions

- **Walk-forward evaluation:** all four conditions are evaluated on the exact same
  188 trading days (Jul 2025 – Apr 2026), ensuring fair directional-accuracy comparison.
- **No leakage:** scalers are fitted on the training window only; news windows are
  bounded by the DST-aware 4:00 PM ET market close on each trading day.
- **Incremental caching:** `llm_predict.py` writes one row per day so an interrupted
  run can be safely resumed without re-spending API budget.
- **Retraining cadence:** LSTM is retrained every 20 evaluation days (not every day)
  to keep CPU runtime manageable.

## API notes

- `MASSIVE_API_KEY` — Massive (formerly Polygon.io), free tier.
  News is fetched in a **single bulk call** for the full date range; no per-day rate
  limiting is needed.
- `OPENAI_API_KEY` — OpenAI, model `gpt-5-nano`.
  `temperature` parameter is not supported by this model; default (1) is used.
