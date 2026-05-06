"""
embed_reports.py — Embed structured news reports using a local sentence-transformers model.

Loads data/processed/news_reports.csv and embeds each report locally using
all-MiniLM-L6-v2 (384-dimensional float32 vectors). No API key required.

Output: data/processed/report_embeddings.parquet
Columns: date (index), emb_0 … emb_383

Resume-safe: dates already in the parquet file are skipped.

Usage:
    python src/embed_reports.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

REPORTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "news_reports.csv")
OUT_PATH     = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "report_embeddings.parquet")

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM   = 384
BATCH_SIZE  = 64


def main():
    if not os.path.exists(REPORTS_PATH):
        print(f"ERROR: {REPORTS_PATH} not found. Run generate_reports.py first.")
        sys.exit(1)

    reports = pd.read_csv(REPORTS_PATH, parse_dates=["date"])
    reports["date"] = pd.to_datetime(reports["date"]).dt.normalize()

    # Resume: load existing embeddings
    if os.path.exists(OUT_PATH):
        existing = pd.read_parquet(OUT_PATH)
        existing.index = pd.to_datetime(existing.index).normalize()
        cached_dates = set(existing.index)
        emb_rows = {d: existing.loc[d].values for d in existing.index}
        print(f"Resuming — {len(cached_dates)} dates already embedded.")
    else:
        cached_dates = set()
        emb_rows = {}

    to_embed = reports[~reports["date"].isin(cached_dates)].reset_index(drop=True)
    print(f"Reports to embed: {len(to_embed)}  (of {len(reports)} total)")

    if len(to_embed) == 0:
        print("Nothing to do.")
        return

    print(f"Loading model: {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)

    for batch_start in range(0, len(to_embed), BATCH_SIZE):
        batch  = to_embed.iloc[batch_start : batch_start + BATCH_SIZE]
        texts  = batch["report_text"].tolist()
        dates  = batch["date"].tolist()
        end_idx = min(batch_start + BATCH_SIZE, len(to_embed))

        print(f"  Embedding rows {batch_start+1}–{end_idx} / {len(to_embed)} ...", end=" ", flush=True)

        vectors = model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False)

        for date, vec in zip(dates, vectors):
            emb_rows[date] = vec.astype(np.float32)

        print("OK")

        cols   = [f"emb_{i}" for i in range(EMBED_DIM)]
        df_out = pd.DataFrame.from_dict(
            {d: v for d, v in emb_rows.items()}, orient="index", columns=cols
        )
        df_out.index.name = "date"
        df_out.to_parquet(OUT_PATH)

    print(f"\nDone. {len(emb_rows)} embeddings saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
