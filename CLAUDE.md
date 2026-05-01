# CLAUDE.md — LLM-Augmented Stock Return Prediction

## Project overview

CBS MSc AI & Machine Learning course project comparing four conditions for predicting
next-day AAPL stock return direction over a walk-forward rolling-window evaluation.

| Condition | Name | Method |
|-----------|------|--------|
| **B**  | LSTM baseline  | 2-layer LSTM on OHLCV + technicals → log return → direction |
| **L1** | LLM price-only | GPT-5-nano on last 10 days of price data → up/down |
| **L2** | LLM news-only  | GPT-5-nano on top-5 Massive article summaries → up/down |
| **L3** | LLM price+news | GPT-5-nano on price + news → up/down |

## Results (version-4, completed 2026-05-01)

All four conditions evaluated on the same **188 trading days** (Jul 2025 – Apr 2026).

| Condition | Dir. Accuracy | CW Acc (conf ≥ 0.7) | MAE | RMSE |
|-----------|:------------:|:-------------------:|:---:|:----:|
| B (LSTM)  | 52.7% | — | 0.01148 | 0.01544 |
| L1 (price only) | 46.8% | 50.0% (n=32) | — | — |
| L2 (news only)  | 48.4% | 30.0% (n=10) | — | — |
| L3 (price+news) | **52.7%** | 36.4% (n=11) | — | — |

Random baseline: **50.0%**. No condition meaningfully outperforms random — consistent
with the efficient market hypothesis over this evaluation window.

## Data

| File | Contents | Stats |
|------|----------|-------|
| `data/processed/prices.csv` | AAPL OHLCV + features | 457 rows, Jul 2024 – Apr 2026 |
| `data/processed/news_cache.csv` | Massive/Polygon news | 456 days, 3 491 articles, 98.7% coverage |

After dropping NaN rows from rolling features (first ~19 rows), 438 rows remain.
Warm-up: rows 0–249. Evaluation: rows 250–437 (188 days).

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
│   ├── figures/                    # 7 charts saved by the notebook
│   ├── train_eval_log.txt          # LSTM training log (10 retraining steps)
│   └── llm_log_{L1,L2,L3}.txt     # Per-day LLM prediction logs
├── src/
│   ├── fetch_data.py               # yfinance download + feature engineering
│   ├── fetch_news.py               # Massive/Polygon bulk news fetch + cache
│   ├── model.py                    # LSTMForecaster class definition
│   ├── train_eval.py               # Walk-forward evaluation for condition B
│   └── llm_predict.py              # LLM conditions L1 / L2 / L3
├── notebooks/
│   └── analysis.ipynb              # All charts and metrics (pre-executed)
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
Output: 457 rows, 2024-07-02 to 2026-04-28.

### Step 2 — Fetch news
```bash
python src/fetch_news.py          # bulk fetch (~3500 articles in one API call)
python src/fetch_news.py --force  # re-fetch and overwrite cache
```
Fetches the full date range in one call (no per-day rate limiting needed).
Saves `data/processed/news_cache.csv` with: `text`, `article_count`, `no_news`,
`aapl_sentiments` (raw labels), `sentiment_score` (numeric mean in [-1, 1]).

### Step 3 — Train and evaluate condition B (LSTM)
```bash
python src/train_eval.py --eval-days 20   # quick test (first 20 eval days)
python src/train_eval.py                  # full run (188 eval days, ~2 min on CPU)
```
Saves `results/predictions_B.csv`.

### Step 4 — Evaluate LLM conditions
```bash
# Dry-run first to verify prompts
python src/llm_predict.py --condition L1 --dry-run

# Small live test (5 days)
python src/llm_predict.py --condition L1 --days 5

# Full runs — resume-safe (skips already-cached dates)
python src/llm_predict.py --condition L1
python src/llm_predict.py --condition L2
python src/llm_predict.py --condition L3
```
Saves `results/predictions_{L1,L2,L3}.csv`. Each run takes ~5–10 min.

### Step 5 — Analysis notebook
```bash
jupyter notebook notebooks/analysis.ipynb
```
Run all cells top-to-bottom. Figures are saved to `results/figures/`.

## Key design decisions

- **Walk-forward evaluation:** all four conditions are evaluated on the exact same
  188 trading days, ensuring fair directional-accuracy comparison.
- **No leakage:** scalers are fitted on the training window only; news windows are
  bounded by the DST-aware 4:00 PM ET market close on each trading day.
- **Incremental caching:** `llm_predict.py` writes one row per day so an interrupted
  run can be safely resumed without re-spending API budget.
- **Retraining cadence:** LSTM is retrained every 20 evaluation days (not every day)
  to keep CPU runtime manageable. 10 retraining steps total.
- **Bulk news fetch:** all 3 491 articles are fetched in one API call and assigned
  to trading days offline; no per-day sleeping required.

## Known quirks

- **Date format inconsistency:** `predictions_B.csv` stores dates as `YYYY-MM-DD`;
  the LLM CSVs store them as `YYYY-MM-DD HH:MM:SS` (artifact of `pd.concat` with
  a cached DataFrame). The notebook's `load_preds()` normalises both with `.str[:10]`.
- **gpt-5-nano temperature:** this model does not support `temperature=0`; only the
  default (1) is accepted. LLM predictions therefore carry sampling variance.
- **Evaluation start date:** the 250-day warm-up over data starting Jul 2024 places
  the first evaluation day at 2025-07-30.

## API notes

- `MASSIVE_API_KEY` — Massive (formerly Polygon.io), free tier.
  News fetched in a single bulk call; no per-day rate limiting needed.
- `OPENAI_API_KEY` — OpenAI, model `gpt-5-nano` (available models as of 2026-05:
  `gpt-5-nano`, `gpt-5-mini`, `gpt-5.2`).
