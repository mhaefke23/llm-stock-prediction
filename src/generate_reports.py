"""
generate_reports.py — Generate structured daily news reports for all price dates.

For each trading day in prices.csv, calls GPT-5-nano with the news text from
news_cache.csv to produce a fixed-format structured report. Days with no news
receive "No news available today." as input.

Output: data/processed/news_reports.csv  (columns: date, report_text)
Resume-safe: dates already in the output file are skipped.

Usage:
    python src/generate_reports.py           # full run (~438 API calls)
    python src/generate_reports.py --days 5  # first 5 dates only (for testing)
"""

import os
import sys
import argparse
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

PRICES_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "prices.csv")
NEWS_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "news_cache.csv")
OUT_PATH     = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "news_reports.csv")

MODEL = "gpt-5-nano"

SYSTEM_PROMPT = (
    "You are a financial analyst. Read all provided news summaries about Apple (AAPL) "
    "and synthesise them into ONE single consolidated report. "
    "Respond strictly in the requested format with no extra text or explanation."
)

REPORT_TEMPLATE = (
    "NEWS SUMMARIES:\n{text}\n\n"
    "Synthesise ALL of the above articles into ONE single report. "
    "Respond in this exact format with no deviations:\n"
    "PRICE_MOMENTUM: [strong upward / mild upward / neutral / mild downward / strong downward]\n"
    "NEWS_SENTIMENT: [strongly positive / positive / neutral / negative / strongly negative]\n"
    "KEY_EVENT: [earnings / product launch / macro / legal / analyst / none]\n"
    "EVENT_IMPACT: [high / medium / low / none]\n"
    "ANALYST_TONE: [bullish / neutral / bearish]\n"
    "SUMMARY: [one sentence, max 20 words, factual]"
)


def build_prompt(news_text: str) -> str:
    return REPORT_TEMPLATE.format(text=news_text)


def call_api(client: OpenAI, prompt: str) -> str | None:
    """Return the raw report string, or None on failure."""
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:
            if attempt == 0:
                print(f"\n    (attempt 1 failed: {exc} — retrying)")
            else:
                print(f"\n    (attempt 2 failed: {exc} — skipping)")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=None,
                        help="Only process the first N price dates (for testing)")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    # Load prices to get the canonical date list (post-dropna)
    prices_df = pd.read_csv(PRICES_PATH, parse_dates=["date"], index_col="date")
    prices_df = prices_df.dropna(subset=["ma5", "ma20", "volatility20", "target"])
    all_dates = prices_df.index.normalize()

    # Load news cache
    news_df = pd.read_csv(NEWS_PATH, parse_dates=["date"])
    news_df["date"] = news_df["date"].dt.normalize()
    news_df = news_df.set_index("date")

    # Resume: load existing reports
    if os.path.exists(OUT_PATH):
        cached = pd.read_csv(OUT_PATH, parse_dates=["date"])
        cached["date"] = pd.to_datetime(cached["date"]).dt.normalize()
        cached_dates = set(cached["date"])
        rows = cached.to_dict("records")
        print(f"Resuming — {len(cached_dates)} dates already cached.")
    else:
        cached_dates = set()
        rows = []

    dates_to_run = [d for d in all_dates if d not in cached_dates]
    if args.days:
        dates_to_run = dates_to_run[: args.days]

    print(f"Dates to process: {len(dates_to_run)}  (of {len(all_dates)} total)")

    null_count = 0
    for i, date in enumerate(dates_to_run):
        print(f"  [{i+1:4d}/{len(dates_to_run)}] {date.date()}", end=" ... ", flush=True)

        # Look up news for this date
        if date in news_df.index and not bool(news_df.loc[date, "no_news"]):
            raw_text = str(news_df.loc[date, "text"]).replace(" | ", "\n")
        else:
            raw_text = "No news available today."

        report = call_api(client, build_prompt(raw_text))
        if report is None:
            print("NULL — skipped")
            null_count += 1
            continue

        # Strip brackets the model sometimes keeps from the format template
        import re
        report = re.sub(r'\[([^\]]+)\]', r'\1', report)

        print("OK")
        rows.append({"date": date.date(), "report_text": report})

        # Write after every day so a crash loses at most one report
        pd.DataFrame(rows).to_csv(OUT_PATH, index=False)

    print(f"\nDone. {len(rows)} reports saved → {OUT_PATH}")
    if null_count:
        print(f"Null/skipped: {null_count}")


if __name__ == "__main__":
    main()
