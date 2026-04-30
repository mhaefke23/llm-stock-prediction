"""
fetch_news.py — Fetch all AAPL news from Massive (Polygon) API in one bulk call,
then assign articles to trading days based on DST-aware market-close windows.

For each trading day t, articles are attributed to t if:
    market_close(t-1) <= published_utc < market_close(t)
where market close = 4:00 PM ET (20:00 UTC during EDT, 21:00 UTC during EST).

Outputs: data/processed/news_cache.csv
Columns: date, text, article_count, no_news

Usage:
    python src/fetch_news.py           # fetch and cache
    python src/fetch_news.py --force   # re-fetch even if cache exists
"""

import os
import sys
import argparse
from datetime import datetime

import pandas as pd
import pytz
from dotenv import load_dotenv
from polygon import RESTClient

# ── Config ───────────────────────────────────────────────────────────────────
TICKER      = "AAPL"
PRICES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "prices.csv")
CACHE_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "news_cache.csv")
TOP_N       = 5
NY_TZ       = pytz.timezone("America/New_York")


def market_close_utc(date: pd.Timestamp) -> datetime:
    """Return 4:00 PM ET on `date` as a UTC-aware datetime (pytz handles DST)."""
    close_et = NY_TZ.localize(datetime(date.year, date.month, date.day, 16, 0, 0))
    return close_et.astimezone(pytz.utc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cache exists")
    args = parser.parse_args()

    if os.path.exists(CACHE_PATH) and not args.force:
        print(f"Cache already exists at {CACHE_PATH}. Use --force to re-fetch.")
        _print_stats()
        return

    # ── API key ──────────────────────────────────────────────────────────────
    load_dotenv()
    api_key = os.getenv("MASSIVE_API_KEY")
    if not api_key:
        print("ERROR: MASSIVE_API_KEY not found in environment / .env file.")
        sys.exit(1)

    # ── Load trading days ────────────────────────────────────────────────────
    prices = pd.read_csv(PRICES_PATH, parse_dates=["date"], index_col="date")
    trading_days = prices.index.sort_values()

    # Build UTC window boundaries for every trading day
    # cutoffs[i] = market close UTC for trading_days[i]
    cutoffs = [market_close_utc(d) for d in trading_days]

    # Fetch window spans the first cutoff to the last cutoff
    fetch_start = cutoffs[0].strftime("%Y-%m-%dT%H:%M:%SZ")
    fetch_end   = cutoffs[-1].strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── Bulk fetch ───────────────────────────────────────────────────────────
    print(f"Fetching all {TICKER} news from {fetch_start} to {fetch_end} ...")
    client = RESTClient(api_key=api_key)

    articles = []
    for item in client.list_ticker_news(
        ticker=TICKER,
        published_utc_gte=fetch_start,
        published_utc_lt=fetch_end,
        sort="published_utc",
        order="desc",
        limit=1000,
    ):
        # Extract the AAPL-specific insight (articles cover multiple tickers)
        aapl_insight = next(
            (ins for ins in (item.insights or []) if ins.ticker == TICKER),
            None
        )
        articles.append({
            "published_utc":        item.published_utc,
            "description":          (item.description or "").strip(),
            "sentiment":            aapl_insight.sentiment if aapl_insight else None,
            "sentiment_reasoning":  aapl_insight.sentiment_reasoning if aapl_insight else None,
        })

    print(f"Total articles fetched: {len(articles)}")

    if not articles:
        print("WARNING: No articles returned. Check API key and date range.")

    raw_df = pd.DataFrame(articles)
    raw_df["published_utc"] = pd.to_datetime(raw_df["published_utc"], utc=True)

    # ── Assign articles to trading days ──────────────────────────────────────
    # Day t receives articles where cutoffs[t-1] <= published_utc < cutoffs[t]
    rows = []
    for i in range(1, len(trading_days)):
        date     = trading_days[i]
        win_lo   = cutoffs[i - 1]
        win_hi   = cutoffs[i]

        mask = (raw_df["published_utc"] >= win_lo) & (raw_df["published_utc"] < win_hi)
        day_articles = raw_df.loc[mask & (raw_df["description"] != "")]

        # Already sorted desc by published_utc from the API; take top N
        day_articles = day_articles.head(TOP_N)

        if len(day_articles) > 0:
            text = " | ".join(day_articles["description"].tolist())

            # Sentiment: collect raw labels and compute a numeric score
            # positive=+1, neutral=0, negative=-1; None → excluded from mean
            score_map = {"positive": 1, "neutral": 0, "negative": -1}
            sentiments   = day_articles["sentiment"].tolist()          # may contain None
            scores       = [score_map[s] for s in sentiments if s in score_map]
            sent_labels  = ",".join(s if s else "" for s in sentiments)
            sent_score   = sum(scores) / len(scores) if scores else None

            rows.append({
                "date":             date.date(),
                "text":             text,
                "article_count":    len(day_articles),
                "no_news":          False,
                "aapl_sentiments":  sent_labels,   # e.g. "positive,neutral,positive"
                "sentiment_score":  sent_score,    # numeric mean in [-1, 1]
            })
        else:
            rows.append({
                "date":             date.date(),
                "text":             "",
                "article_count":    0,
                "no_news":          True,
                "aapl_sentiments":  None,
                "sentiment_score":  None,
            })

    cache_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    cache_df.to_csv(CACHE_PATH, index=False)
    print(f"Saved → {CACHE_PATH}")

    _print_stats()


def _print_stats():
    df = pd.read_csv(CACHE_PATH)
    total       = len(df)
    with_news   = int((~df["no_news"]).sum())
    no_news_cnt = int(df["no_news"].sum())
    print(f"\n── Coverage stats ──────────────────────────────")
    print(f"Total trading days cached : {total}")
    print(f"Days with articles        : {with_news}  ({100*with_news/total:.1f}%)")
    print(f"Days with no_news=True    : {no_news_cnt}  ({100*no_news_cnt/total:.1f}%)")
    if with_news:
        avg = df.loc[~df["no_news"], "article_count"].mean()
        print(f"Avg articles on news days : {avg:.1f}")


if __name__ == "__main__":
    main()
