"""
llm_features.py
---------------
For each trading day in prices.csv:
  1. Fetches pre-close AAPL news from Alpha Vantage (monthly batches, cached).
  2. Filters articles to the leakage-safe window:
       prev_trading_day 16:00 ET  <  publish_time  <=  today 16:00 ET
  3. Builds a prompt with the last 5 days of prices + today's headlines.
  4. Calls Ollama (llama3.2) locally to produce structured JSON:
       { "summary": "...", "direction": "up"|"down", "confidence": 0.0–1.0 }
  5. Appends each result to data/processed/llm_features.csv immediately
     so interrupted runs can be resumed without re-calling the LLM.

Usage:
  python src/llm_features.py               # full run
  python src/llm_features.py --dry-run     # skip API + LLM calls, preview prompts
  python src/llm_features.py --days 3      # process only 3 unprocessed days (for testing)
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import ollama
import pandas as pd
import pytz
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
TICKER      = "AAPL"
MODEL       = "llama3.2"
AV_KEY      = os.getenv("ALPHAVANTAGE_API_KEY")
ET          = pytz.timezone("America/New_York")
UTC         = pytz.utc
CLOSE_HOUR  = 16  # 4 pm ET market close

PRICES_CSV  = Path("data/processed/prices.csv")
RAW_NEWS    = Path("data/raw/av_news_raw.json")
LLM_OUT     = Path("data/processed/llm_features.csv")

# ── Phase 1: Fetch & cache news ───────────────────────────────────────────────

def _av_request(time_from: str, time_to: str) -> list:
    """Single Alpha Vantage NEWS_SENTIMENT request; returns the feed list."""
    url = (
        "https://www.alphavantage.co/query"
        f"?function=NEWS_SENTIMENT&tickers={TICKER}"
        f"&time_from={time_from}&time_to={time_to}"
        f"&limit=200&apikey={AV_KEY}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "feed" not in data:
        print(f"  Warning — unexpected AV response keys: {list(data.keys())}")
        if "Information" in data:
            print(f"  AV message: {data['Information']}")
        return []
    return data["feed"]


def fetch_all_news(start_date, end_date, dry_run: bool) -> list:
    """
    Fetch all AAPL news for the date range in monthly batches (≈12 API calls).
    Caches the full result to RAW_NEWS; re-uses the cache on subsequent runs.
    Alpha Vantage free tier: 25 req/day, 5 req/min — we sleep 13s between calls.
    """
    if RAW_NEWS.exists():
        print(f"Loading cached news from {RAW_NEWS} …")
        return json.loads(RAW_NEWS.read_text())

    if dry_run:
        print("[DRY RUN] Would fetch news from Alpha Vantage — skipping.")
        return []

    articles = []
    current = start_date.replace(day=1)

    while current <= end_date:
        # End of current month (capped at end_date)
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1, day=1)
        else:
            next_month = current.replace(month=current.month + 1, day=1)
        month_end = min(next_month - timedelta(days=1), end_date)

        t_from = current.strftime("%Y%m%dT0000")
        t_to   = month_end.strftime("%Y%m%dT2359")

        print(f"  Fetching {t_from} → {t_to} … ", end="", flush=True)
        batch = _av_request(t_from, t_to)
        articles.extend(batch)
        print(f"{len(batch)} articles")

        current = next_month
        time.sleep(13)  # stay under 5 req/min on the free tier

    RAW_NEWS.parent.mkdir(parents=True, exist_ok=True)
    RAW_NEWS.write_text(json.dumps(articles, indent=2))
    print(f"Saved {len(articles)} total articles → {RAW_NEWS}")
    return articles


# ── Phase 2: Assign articles to trading days ──────────────────────────────────

def _parse_av_timestamp(ts_str: str) -> datetime:
    """Parse Alpha Vantage timestamp (YYYYMMDDTHHMMSS, UTC) → ET-aware datetime."""
    dt = datetime.strptime(ts_str, "%Y%m%dT%H%M%S")
    return UTC.localize(dt).astimezone(ET)


def articles_for_day(all_articles: list, trading_day, prev_trading_day) -> list:
    """
    Leakage-safe filter: return only articles published in the window
      (prev_trading_day 16:00 ET,  trading_day 16:00 ET].
    This ensures predictions use only information available before today's close.
    """
    window_start = ET.localize(datetime(prev_trading_day.year, prev_trading_day.month,
                                        prev_trading_day.day, CLOSE_HOUR))
    window_end   = ET.localize(datetime(trading_day.year, trading_day.month,
                                        trading_day.day, CLOSE_HOUR))
    result = []
    for art in all_articles:
        ts_str = art.get("time_published", "")
        if not ts_str:
            continue
        try:
            ts = _parse_av_timestamp(ts_str)
        except ValueError:
            continue
        if window_start < ts <= window_end:
            result.append(art)
    return result


# ── Phase 3: Build prompt ─────────────────────────────────────────────────────

def build_prompt(trading_day, price_window: pd.DataFrame, articles: list) -> str:
    """
    Construct the LLM prompt from the last 5 days of OHLCV features
    and today's pre-close headlines.
    """
    # Price history block
    price_lines = []
    for idx, row in price_window.iterrows():
        sign = "+" if row["daily_return"] >= 0 else ""
        price_lines.append(
            f"  {idx.date()}  close=${row['close']:.2f}  "
            f"return={sign}{row['daily_return'] * 100:.2f}%  "
            f"ma5=${row['ma5']:.2f}  ma20=${row['ma20']:.2f}"
        )

    # Headlines block — cap at 5 articles to keep the prompt concise
    if articles:
        headline_lines = []
        for i, art in enumerate(articles[:5], 1):
            try:
                ts_et = _parse_av_timestamp(art["time_published"])
                time_str = ts_et.strftime("%H:%M ET")
            except (ValueError, KeyError):
                time_str = "??:?? ET"
            headline_lines.append(f"  {i}. [{time_str}] {art['title']}")
        headlines_block = "\n".join(headline_lines)
    else:
        headlines_block = "  (No news articles found for this trading day.)"

    return (
        f"You are a financial analyst assistant. Analyze the data below for AAPL on {trading_day}.\n\n"
        f"PRICE HISTORY — last 5 trading days:\n"
        + "\n".join(price_lines)
        + f"\n\nNEWS HEADLINES published before {trading_day} market close (16:00 ET):\n"
        + headlines_block
        + '\n\nRespond ONLY with a valid JSON object, no other text:\n'
        '{\n'
        '  "summary": "<2-3 sentence qualitative assessment of market sentiment and price momentum>",\n'
        '  "direction": "<up or down — your prediction for tomorrow\'s AAPL return direction>",\n'
        '  "confidence": <float 0.0-1.0, where 1.0 = very confident>\n'
        '}'
    )


# ── Phase 4: Call Ollama ──────────────────────────────────────────────────────

_JSON_RE  = re.compile(r"\{.*\}", re.DOTALL)
_FALLBACK = {"summary": "LLM response could not be parsed.", "direction": "up", "confidence": 0.5}


def call_ollama(prompt: str, retries: int = 3) -> dict:
    """
    Send the prompt to local Ollama (llama3.2) with JSON mode enabled.
    Retries up to `retries` times on parse failure.
    Returns a fallback dict if all attempts fail.
    """
    for attempt in range(retries):
        content = None
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.1},
            )
            # Support both object-style (.message.content) and dict-style access
            content = (
                response.message.content
                if hasattr(response, "message")
                else response["message"]["content"]
            )
            return json.loads(content)

        except (json.JSONDecodeError, KeyError, AttributeError) as e:
            print(f"\n    Parse attempt {attempt + 1} failed ({e})", end="")
            # Try extracting a JSON substring from the raw content
            if content:
                match = _JSON_RE.search(content)
                if match:
                    try:
                        return json.loads(match.group())
                    except json.JSONDecodeError:
                        pass

        except Exception as e:
            print(f"\n    Ollama call failed (attempt {attempt + 1}): {e}", end="")

    print("\n    All attempts failed — using fallback values.")
    return _FALLBACK.copy()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate LLM features for each trading day.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip API and LLM calls; print prompts instead.")
    parser.add_argument("--days", type=int, default=None,
                        help="Stop after processing N new days (useful for testing).")
    args = parser.parse_args()

    # Load price data
    prices = pd.read_csv(PRICES_CSV, index_col="date", parse_dates=True)
    trading_days = list(prices.index)
    print(f"Loaded {len(trading_days)} trading days from {PRICES_CSV}")

    # Phase 1: fetch and cache all news articles
    start_date   = trading_days[0].date()
    end_date     = trading_days[-1].date()
    all_articles = fetch_all_news(start_date, end_date, dry_run=args.dry_run)
    print(f"Total articles available: {len(all_articles)}")

    # Load already-processed days so interrupted runs can resume
    LLM_OUT.parent.mkdir(parents=True, exist_ok=True)
    if LLM_OUT.exists():
        done_dates = set(pd.read_csv(LLM_OUT, usecols=["date"])["date"].astype(str).tolist())
        print(f"Resuming — {len(done_dates)} days already processed.")
    else:
        done_dates = set()
        # Write CSV header once
        pd.DataFrame(columns=["date", "summary", "direction", "confidence"]).to_csv(LLM_OUT, index=False)

    # Phases 2–4: process each trading day
    processed_count = 0
    for i, day in enumerate(trading_days):
        day_str = str(day.date())

        if day_str in done_dates:
            continue
        if i < 5:
            # Need at least 5 prior rows for the price window
            continue
        if args.days is not None and processed_count >= args.days:
            print(f"Reached --days limit ({args.days}). Stopping.")
            break

        prev_day     = trading_days[i - 1]
        price_window = prices.iloc[i - 5 : i]  # 5 days before today, not including today

        articles = articles_for_day(all_articles, day.date(), prev_day.date())
        print(f"[{i+1:>3}/{len(trading_days)}] {day_str}  ({len(articles):>2} articles) … ", end="", flush=True)

        prompt = build_prompt(day.date(), price_window, articles)

        if args.dry_run:
            print()
            print("─" * 64)
            print(prompt)
            print("─" * 64)
            result = {"summary": "[DRY RUN — no LLM called]", "direction": "up", "confidence": 0.5}
        else:
            result = call_ollama(prompt)
            print(f"{result.get('direction', '?'):>4} ({result.get('confidence', 0.5):.2f})")

        # Validate and clamp fields before saving
        direction  = str(result.get("direction", "up")).lower().strip()
        direction  = direction if direction in ("up", "down") else "up"
        confidence = max(0.0, min(1.0, float(result.get("confidence", 0.5))))
        summary    = str(result.get("summary", "")).replace("\n", " ").strip()

        # Append row immediately (enables resuming if the process is interrupted)
        pd.DataFrame([{
            "date":       day_str,
            "summary":    summary,
            "direction":  direction,
            "confidence": confidence,
        }]).to_csv(LLM_OUT, mode="a", header=False, index=False)

        processed_count += 1

    # Summary
    print(f"\nDone. LLM features saved → {LLM_OUT}")
    df = pd.read_csv(LLM_OUT)
    print(f"Total rows: {len(df)}")
    if not df.empty:
        print(df.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
