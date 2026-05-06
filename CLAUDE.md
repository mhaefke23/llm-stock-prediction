# CLAUDE.md — LLM-Augmented Stock Return Prediction

## Project overview

CBS MSc AI & Machine Learning course project comparing 11 conditions for predicting
next-day AAPL stock return direction over a walk-forward rolling-window evaluation.

| Condition | Name | Method |
|-----------|------|--------|
| **B**  | LSTM regressor | 2-layer LSTM on OHLCV + technicals → log return → direction |
| **B1** | LSTM classifier (single-day) | LSTM with BCELoss, seq_len=1 → direction |
| **B2** | LSTM classifier (sequence) | LSTM with BCELoss, seq_len=20 → direction |
| **L1** | LLM price-only | GPT-5-nano on last 10 days of price data → up/down |
| **L2** | LLM news-only  | GPT-5-nano on top-5 Massive article summaries → up/down |
| **L3** | LLM price+news | GPT-5-nano on price + news → up/down |
| **X1** | XGB regressor (single-day) | XGBoost on current-day features → log return → direction |
| **X2** | XGB regressor (20-day lags) | XGBoost on flattened 20-day features → log return → direction |
| **X3** | XGB classifier (single-day) | XGBoost on current-day features → direction |
| **X4** | XGB classifier (20-day lags) | XGBoost on flattened 20-day features → direction |
| **XN** | XGB classifier + news | X3 + PCA-reduced LLM report embeddings → direction |

## Results (version-5, completed 2026-05-06)

All 11 conditions evaluated on the same **188 trading days** (Jul 2025 – Apr 2026).

| Condition | Dir. Accuracy | Pred. "up" rate | MAE | RMSE |
|-----------|:------------:|:---------------:|:---:|:----:|
| B — LSTM regressor | 52.7% | 47.9% | 0.01148 | 0.01544 |
| B1 — LSTM clf (single-day) | 52.1% | 42.0% | — | — |
| B2 — LSTM clf (sequence) | 50.0% | 51.6% | — | — |
| L1 — LLM price only | 46.8% | 63.3% | — | — |
| L2 — LLM news only | 48.4% | 68.1% | — | — |
| L3 — LLM price+news | 52.7% | 64.9% | — | — |
| X1 — XGB regressor (single-day) | 45.2% | 47.9% | 0.01061 | 0.01490 |
| X2 — XGB regressor (20-day lags) | 46.8% | 53.7% | 0.01057 | 0.01495 |
| **X3 — XGB clf (single-day)** | **56.4%** | 72.9% | — | — |
| X4 — XGB clf (20-day lags) | 53.7% | 87.2% | — | — |
| XN — XGB clf + news | 52.7% | 84.0% | — | — |

Random baseline: **50.0%**. Actual market up-rate: **51.6%**.

Key findings: XGB classifiers outperform XGB regressors significantly (framing the
task as classification avoids noisy sign estimation of near-zero returns). High
predicted "up" rates for X3/X4/XN inflate accuracy — genuine directional skill is
limited. News does not help in either the LLM or XGBoost family.

## Data

| File | Contents | Stats |
|------|----------|-------|
| `data/processed/prices.csv` | AAPL OHLCV + features | 457 rows, Jul 2024 – Apr 2026 |
| `data/processed/news_cache.csv` | Massive/Polygon news | 456 days, 3 491 articles, 98.7% coverage |
| `data/processed/news_reports.csv` | GPT-5-nano structured daily reports | one row per trading day |

After dropping NaN rows from rolling features (first ~19 rows), 438 rows remain.
Warm-up: rows 0–249. Evaluation: rows 250–437 (188 days).

## Repository layout

```
llm-stock-prediction/
├── data/
│   ├── raw/                        # (empty — no raw files committed)
│   └── processed/
│       ├── prices.csv              # AAPL OHLCV + features, Jul 2024 – Apr 2026
│       ├── news_cache.csv          # Massive/Polygon news, one row per trading day
│       ├── news_reports.csv        # GPT-5-nano structured daily reports
│       └── report_embeddings.parquet  # sentence-transformer embeddings (gitignored if large)
├── results/
│   ├── predictions_B.csv           # LSTM regressor predictions
│   ├── predictions_B1.csv          # LSTM classifier (seq_len=1) predictions
│   ├── predictions_B2.csv          # LSTM classifier (seq_len=20) predictions
│   ├── predictions_L1.csv          # LLM price-only predictions
│   ├── predictions_L2.csv          # LLM news-only predictions
│   ├── predictions_L3.csv          # LLM price+news predictions
│   ├── predictions_X1.csv          # XGB regressor (single-day) predictions
│   ├── predictions_X2.csv          # XGB regressor (20-day lags) predictions
│   ├── predictions_X3.csv          # XGB classifier (single-day) predictions
│   ├── predictions_X4.csv          # XGB classifier (20-day lags) predictions
│   ├── predictions_XN.csv          # XGB classifier + news predictions
│   ├── figures/                    # Charts saved by the notebook
│   ├── train_eval_log.txt          # LSTM training log (10 retraining steps)
│   └── llm_log_{L1,L2,L3}.txt     # Per-day LLM prediction logs
├── src/
│   ├── fetch_data.py               # yfinance download + feature engineering
│   ├── fetch_news.py               # Massive/Polygon bulk news fetch + cache
│   ├── generate_reports.py         # GPT-5-nano structured daily report generation
│   ├── embed_reports.py            # Sentence-transformer embedding of reports
│   ├── model.py                    # LSTMForecaster class definition
│   ├── train_eval.py               # Walk-forward evaluation for condition B
│   ├── train_eval_lstm_clf.py      # Walk-forward evaluation for conditions B1, B2
│   ├── train_eval_xgb.py           # Walk-forward evaluation for conditions X1–X4
│   ├── train_eval_xgb_news.py      # Walk-forward evaluation for condition XN
│   └── llm_predict.py              # LLM conditions L1 / L2 / L3
├── notebooks/
│   ├── analysis.ipynb              # All charts and metrics (pre-executed)
│   └── analysis.pdf                # Exported PDF report
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
Saves `data/processed/news_cache.csv`.

### Step 3 — Generate and embed news reports (required for XN)
```bash
python src/generate_reports.py    # GPT-5-nano structured reports (~188 API calls)
python src/embed_reports.py       # local sentence-transformer embeddings
```
Saves `data/processed/news_reports.csv` and `data/processed/report_embeddings.parquet`.
Both scripts are resume-safe (skip already-processed dates).

### Step 4 — Train and evaluate LSTM conditions
```bash
python src/train_eval.py --eval-days 20        # quick test
python src/train_eval.py                        # full run — condition B
python src/train_eval_lstm_clf.py --condition B1
python src/train_eval_lstm_clf.py --condition B2
```
Saves `results/predictions_{B,B1,B2}.csv`.

### Step 5 — Train and evaluate XGBoost conditions
```bash
python src/train_eval_xgb.py --condition X1    # or X2, X3, X4
python src/train_eval_xgb.py --condition X3 --eval-days 20  # quick test
python src/train_eval_xgb_news.py              # condition XN
```
Saves `results/predictions_{X1,X2,X3,X4,XN}.csv`.

### Step 6 — Evaluate LLM conditions
```bash
python src/llm_predict.py --condition L1 --dry-run   # verify prompts
python src/llm_predict.py --condition L1
python src/llm_predict.py --condition L2
python src/llm_predict.py --condition L3
```
Saves `results/predictions_{L1,L2,L3}.csv`. Each run takes ~5–10 min.

### Step 7 — Analysis notebook
```bash
jupyter notebook notebooks/analysis.ipynb
```
Run all cells top-to-bottom. Figures saved to `results/figures/`.
Export PDF: `jupyter nbconvert --to pdf --no-input notebooks/analysis.ipynb`
(requires XeLaTeX — see BasicTeX install instructions).

## Key design decisions

- **Walk-forward evaluation:** all 11 conditions are evaluated on the exact same
  188 trading days, ensuring fair directional-accuracy comparison.
- **No leakage:** scalers and PCA fitted on training window only; news windows are
  bounded by the DST-aware 4:00 PM ET market close on each trading day.
- **Classification vs regression:** XGB conditions include both regressor (derive
  direction from sign of predicted return) and classifier (direct binary target)
  variants to test the effect of task framing.
- **Incremental caching:** `llm_predict.py` and `generate_reports.py` write one row
  per day so interrupted runs resume without re-spending API budget.
- **Retraining cadence:** LSTM and XGBoost models retrain every 20 evaluation days
  on a rolling 250-day window. 10 retraining steps total.
- **Bulk news fetch:** all 3 491 articles fetched in one API call and assigned to
  trading days offline; no per-day sleeping required.

## Known quirks

- **Date format inconsistency:** `predictions_B.csv` stores dates as `YYYY-MM-DD`;
  the LLM CSVs store them as `YYYY-MM-DD HH:MM:SS` (artifact of `pd.concat` with
  a cached DataFrame). The notebook's `load_preds()` normalises both with `.str[:10]`.
- **gpt-5-nano temperature:** this model does not support `temperature=0`; only the
  default (1) is accepted. LLM predictions therefore carry sampling variance.
- **Evaluation start date:** the 250-day warm-up over data starting Jul 2024 places
  the first evaluation day at 2025-07-30.
- **XGB upward bias:** XGB classifiers (X3, X4, XN) predict "up" on 73–87% of days
  despite the market rising on only 51.6%, due to class imbalance in rolling training
  windows during a broadly trending market.

## API notes

- `MASSIVE_API_KEY` — Massive (formerly Polygon.io), free tier.
  News fetched in a single bulk call; no per-day rate limiting needed.
- `OPENAI_API_KEY` — OpenAI, model `gpt-5-nano` (available models as of 2026-05:
  `gpt-5-nano`, `gpt-5-mini`, `gpt-5.2`).
