# LLM-Augmented Stock Return Prediction

A course project for the AI & Machine Learning course at Copenhagen Business School (MSc Information Systems).

Compares three LSTM-based models for predicting next-day AAPL log returns, progressively adding LLM-generated features to a price-only baseline.

## Conditions

| | C1 — Baseline | C2 — LLM Embeddings | C3 — LLM Predictor |
|---|---|---|---|
| OHLCV + technical indicators | ✓ | ✓ | ✓ |
| Sentence embeddings of news summaries (PCA-8) | | ✓ | ✓ |
| LLM direction prediction + confidence score | | | ✓ |
| **Input dimensions** | **9** | **17** | **19** |

## Results

| | MAE | RMSE | R² | Directional Acc |
|---|---|---|---|---|
| C1 — Baseline | 0.0099 | 0.0131 | -0.08 | 53.3% |
| C2 — LLM Embeddings | 0.0102 | 0.0136 | -0.17 | **66.7%** |
| C3 — LLM Predictor | 0.0132 | 0.0166 | -0.75 | 33.3% |

C2 outperforms the baseline on directional accuracy (the most practically meaningful metric), suggesting LLM-generated news summaries carry predictive signal beyond price data alone.

## Pipeline

```
fetch_data.py       → data/processed/prices.csv
llm_features.py     → data/processed/llm_features.csv
features.py         → data/processed/features_c1/c2/c3.csv
train.py            → results/
notebooks/analysis.ipynb  ← front-end: full walkthrough + visualizations
```

1. **`fetch_data.py`** — Downloads 1 year of AAPL OHLCV via yfinance, computes log returns, moving averages (MA-5, MA-20), and rolling volatility.
2. **`llm_features.py`** — Fetches AAPL news from Alpha Vantage (cached), then calls a local Ollama LLM (`llama3.2`) for each trading day to produce a qualitative summary, direction prediction (up/down), and confidence score. Leakage-safe: only uses articles published before 16:00 ET on day *t*.
3. **`features.py`** — Embeds summaries with `all-MiniLM-L6-v2`, reduces to 8 dims via PCA, and assembles the three feature sets. All preprocessing fitted on training data only.
4. **`model.py`** — 2-layer LSTM (hidden=64) with a fully connected output layer.
5. **`train.py`** — Trains one model per condition with early stopping, evaluates on the held-out test set, saves predictions and metrics.

## Setup

**Prerequisites:** [Ollama](https://ollama.com) running locally with `llama3.2` pulled.

```bash
git clone https://github.com/mhaefke23/llm-stock-prediction.git
cd llm-stock-prediction
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
ALPHAVANTAGE_API_KEY=your_key_here
```

## Running the pipeline

```bash
# 1. Fetch price data
python src/fetch_data.py

# 2. Generate LLM features (resumable; use --dry-run to preview, --days N to test)
python src/llm_features.py

# 3. Build feature sets
python src/features.py

# 4. Train and evaluate
python src/train.py
```

Then open `notebooks/analysis.ipynb` for the full walkthrough and visualizations.

## Data

- **Price data:** 1 year of AAPL daily OHLCV via [yfinance](https://github.com/ranaroussi/yfinance)
- **News:** [Alpha Vantage News Sentiment API](https://www.alphavantage.co/documentation/#news-sentiment) (free tier, ~50 articles/month)
- **LLM:** [Ollama](https://ollama.com) running `llama3.2` (3B) locally — no API key required
- **Embeddings:** `all-MiniLM-L6-v2` via [sentence-transformers](https://www.sbert.net)
- **Split:** 70% train / 15% val / 15% test, strictly by time
