"""
llm_predict.py — LLM-based next-day AAPL direction prediction (conditions L1, L2, L3).

L1: price data only    (last 10 trading days of OHLCV + technicals)
L2: news only          (top 5 article summaries from Massive/Polygon)
L3: price + news       (both of the above)

Evaluation is restricted to the exact same days as condition B (predictions_B.csv)
to ensure fair comparison across all four conditions.

Outputs: results/predictions_{L1,L2,L3}.csv
Columns: date, actual_direction, predicted_direction, confidence, reasoning

Usage:
    python src/llm_predict.py --condition L1 --dry-run    # preview prompts, no API call
    python src/llm_predict.py --condition L1 --days 5     # live test on 5 days
    python src/llm_predict.py --condition L1              # full run
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────
PRICES_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "prices.csv")
NEWS_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "news_cache.csv")
B_PREDS_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "predictions_B.csv")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "results")

MODEL    = "gpt-5-nano"
LOOKBACK = 10   # trading days of price history shown in prompt


# ── Prompt construction ───────────────────────────────────────────────────────

def build_price_table(prices_df: pd.DataFrame, eval_date: pd.Timestamp) -> str:
    """Format the last LOOKBACK trading days up to eval_date as a readable table."""
    loc = prices_df.index.get_loc(eval_date)
    window = prices_df.iloc[max(0, loc - LOOKBACK + 1) : loc + 1]

    lines = ["Date        | Close    | MA5      | MA20     | Daily Ret | Volatility"]
    lines.append("-" * 72)
    for date, row in window.iterrows():
        lines.append(
            f"{str(date.date()):<12}| "
            f"{row['close']:>8.2f} | "
            f"{row['ma5']:>8.2f} | "
            f"{row['ma20']:>8.2f} | "
            f"{row['daily_log_return']:>+9.4f} | "
            f"{row['volatility20']:>10.4f}"
        )
    return "\n".join(lines)


def build_prompt(condition: str, eval_date: pd.Timestamp,
                 prices_df: pd.DataFrame, news_df: pd.DataFrame) -> str:
    """Assemble the full prompt for a given condition and evaluation date."""
    parts = ["You are a financial analyst predicting next-day AAPL stock movement.\n"]

    # Price section — L1 and L3
    if condition in ("L1", "L3"):
        parts.append(
            f"Recent AAPL performance (last {LOOKBACK} trading days):\n"
            + build_price_table(prices_df, eval_date)
            + "\n"
        )

    # News section — L2 and L3
    if condition in ("L2", "L3"):
        date_key = eval_date.date()
        if date_key in news_df.index and not bool(news_df.loc[date_key, "no_news"]):
            # Replace " | " separator with newlines for readability
            news_text = str(news_df.loc[date_key, "text"]).replace(" | ", "\n")
        else:
            news_text = "No news available for this trading day."
        parts.append(f"Today's AAPL news (published before market close):\n{news_text}\n")

    parts.append(
        "Based on the above, predict whether AAPL will close HIGHER or LOWER tomorrow.\n"
        "Respond in JSON only — no other text:\n"
        '{"direction": "up" or "down", "confidence": 0.0-1.0, "reasoning": "max one sentence"}'
    )
    return "\n".join(parts)


# ── API call ──────────────────────────────────────────────────────────────────

def call_api(client: OpenAI, prompt: str) -> dict | None:
    """Call the OpenAI API and parse the JSON response. Retries once on failure."""
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown code fences if the model wraps the JSON
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            parsed = json.loads(raw)
            assert parsed["direction"] in ("up", "down"), "invalid direction"
            assert 0.0 <= float(parsed["confidence"]) <= 1.0, "confidence out of range"
            return {
                "direction":  parsed["direction"],
                "confidence": float(parsed["confidence"]),
                "reasoning":  str(parsed.get("reasoning", "")),
            }
        except Exception as exc:
            if attempt == 0:
                print(f"\n    (attempt 1 failed: {exc} — retrying)")
            else:
                print(f"\n    (attempt 2 failed: {exc} — logging null)")
    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=["L1", "L2", "L3"])
    parser.add_argument("--days",    type=int,      default=None,
                        help="Only run the first N evaluation days")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print prompts for the first 3 days without calling the API")
    args = parser.parse_args()

    load_dotenv()

    # ── Load data ─────────────────────────────────────────────────────────────
    prices_df = pd.read_csv(PRICES_PATH, parse_dates=["date"], index_col="date")
    prices_df = prices_df.dropna(subset=["ma5", "ma20", "volatility20"])

    news_df = pd.read_csv(NEWS_PATH, parse_dates=["date"])
    news_df["date"] = news_df["date"].dt.date
    news_df = news_df.set_index("date")

    # Use exactly the same evaluation days as condition B
    b_preds = pd.read_csv(B_PREDS_PATH, parse_dates=["date"])
    eval_dates  = pd.DatetimeIndex(b_preds["date"])
    actual_dirs = dict(zip(b_preds["date"].dt.date, b_preds["actual_direction"]))

    if args.days:
        eval_dates = eval_dates[: args.days]

    # ── Load existing cache (enables resume after interruption) ───────────────
    out_path = os.path.join(RESULTS_DIR, f"predictions_{args.condition}.csv")
    if os.path.exists(out_path):
        cached_df    = pd.read_csv(out_path, parse_dates=["date"])
        cached_dates = set(cached_df["date"].dt.date)
        print(f"Resuming — {len(cached_dates)} days already cached.")
    else:
        cached_df    = pd.DataFrame()
        cached_dates = set()

    # ── Dry-run: print first 3 prompts and exit ───────────────────────────────
    if args.dry_run:
        for d in eval_dates[:3]:
            print(f"\n{'='*72}")
            print(f"  DRY RUN | condition={args.condition} | date={d.date()}")
            print("="*72)
            print(build_prompt(args.condition, d, prices_df, news_df))
        return

    # ── Live evaluation ───────────────────────────────────────────────────────
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in environment / .env file.")
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    to_run     = [d for d in eval_dates if d.date() not in cached_dates]
    null_dates = []
    new_rows   = []

    print(f"Condition {args.condition} | model={MODEL} | "
          f"{len(to_run)} days to evaluate ({len(cached_dates)} cached)")

    for i, eval_date in enumerate(to_run):
        print(f"  [{i+1:4d}/{len(to_run)}] {eval_date.date()}", end=" ... ", flush=True)

        prompt = build_prompt(args.condition, eval_date, prices_df, news_df)
        result = call_api(client, prompt)

        if result is None:
            print("NULL — skipped")
            null_dates.append(eval_date.date())
            continue

        pred_dir = 1 if result["direction"] == "up" else 0
        new_rows.append({
            "date":                eval_date.date(),
            "actual_direction":    actual_dirs.get(eval_date.date()),
            "predicted_direction": pred_dir,
            "confidence":          result["confidence"],
            "reasoning":           result["reasoning"],
        })
        print(f"{result['direction']}  conf={result['confidence']:.2f}")

        # Write after every day so a crash loses at most one prediction
        all_rows = pd.concat(
            [cached_df, pd.DataFrame(new_rows)], ignore_index=True
        ) if not cached_df.empty else pd.DataFrame(new_rows)
        all_rows.to_csv(out_path, index=False)

    # ── Metrics ───────────────────────────────────────────────────────────────
    if not os.path.exists(out_path):
        print("No predictions saved (all days were null or already cached).")
        return

    final     = pd.read_csv(out_path).dropna(subset=["actual_direction", "predicted_direction"])
    act        = final["actual_direction"].values.astype(int)
    pred       = final["predicted_direction"].values.astype(int)
    conf       = final["confidence"].values

    dir_acc    = np.mean(act == pred)
    high_conf  = final[conf >= 0.7]
    cw_acc     = (
        np.mean(high_conf["actual_direction"].values == high_conf["predicted_direction"].values)
        if len(high_conf) else float("nan")
    )

    print(f"\n── Condition {args.condition} Results ─────────────────────────────")
    print(f"Evaluation days         : {len(final)}")
    print(f"Null / skipped          : {len(null_dates)}")
    print(f"Dir. Accuracy           : {dir_acc:.1%}  (random baseline: 50.0%)")
    print(f"Conf-weighted Acc (≥0.7): {cw_acc:.1%}  ({len(high_conf)} days)")

    if null_dates:
        print(f"\nNull dates logged: {null_dates}")


if __name__ == "__main__":
    main()
