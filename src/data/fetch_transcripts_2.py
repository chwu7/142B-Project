"""
src/data/fetch_transcripts.py
OWNER: Person 1 (Data pipeline)

Loads earnings call transcripts from a local Kaggle dataset directory.
Expects .txt files named like: 2018_Q1_AAPL.txt (case-insensitive ticker).

Instead of hitting the FMP API, this script reads each .txt file,
parses the metadata from the filename, and saves a normalised JSON file
per call to data/raw/transcripts/ — matching the same schema the rest of
the pipeline expects:
    { "symbol", "date", "quarter", "year", "content" }

Usage:
    # Point KAGGLE_TRANSCRIPTS_DIR at your unzipped Kaggle folder, then:
    python src/data/fetch_transcripts.py

    # Or override the source dir inline:
    KAGGLE_TRANSCRIPTS_DIR=/path/to/kaggle python src/data/fetch_transcripts.py
"""

import os
import re
import json
from dotenv import load_dotenv
from src.utils.config import DATE_START, DATE_END, TICKERS, DATA_RAW_DIR

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
# Set KAGGLE_TRANSCRIPTS_DIR in your .env or shell before running.
KAGGLE_DIR = os.getenv("KAGGLE_TRANSCRIPTS_DIR", "src/data/NLP_Dataset")
OUT_DIR    = os.path.join(DATA_RAW_DIR, "transcripts")

# Regex for filenames like: 2018_Q1_AAPL.txt  (year_quarter_ticker)
FILENAME_RE = re.compile(
    r"^(?P<year>\d{4})_(?P<quarter>Q[1-4])_(?P<ticker>[A-Za-z]+)\.txt$",
    re.IGNORECASE,
)

# Approximate quarter-start → date string used as the call date.
# fetch_returns.py will look up real price data around this date anyway,
# so a ~month-level approximation is fine for matching.
QUARTER_TO_MONTH = {"Q1": "03", "Q2": "06", "Q3": "09", "Q4": "12"}


def parse_filename(fname: str) -> dict | None:
    """
    Parses a Kaggle filename into metadata.
    Returns None if the filename doesn't match the expected pattern.
    """
    m = FILENAME_RE.match(fname)
    if not m:
        return None
    year    = m.group("year")
    quarter = m.group("quarter").upper()
    ticker  = m.group("ticker").upper()
    # Use the last month of the quarter as a proxy call date (YYYY-MM-DD)
    month   = QUARTER_TO_MONTH[quarter]
    date    = f"{year}-{month}-01"
    return {"ticker": ticker, "year": year, "quarter": quarter, "date": date}


def load_transcript_files(kaggle_dir: str) -> list[dict]:
    """
    Walks kaggle_dir, parses every matching .txt file, and returns a list of
    transcript dicts with the downstream-expected schema:
        { symbol, date, quarter, year, content }
    """
    if not os.path.isdir(kaggle_dir):
        raise FileNotFoundError(
            f"Kaggle transcript directory not found: {kaggle_dir}\n"
            f"Set KAGGLE_TRANSCRIPTS_DIR in your .env to the correct path."
        )

    transcripts = []
    skipped     = []

    for root, _, files in os.walk(kaggle_dir):
        for fname in files:
            meta = parse_filename(fname)
            if meta is None:
                skipped.append(fname)
                continue

            fpath = os.path.join(root, fname)
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                content = f.read().strip()

            if not content:
                skipped.append(fname)
                continue

            transcripts.append({
                "symbol":  meta["ticker"],
                "date":    meta["date"],
                "quarter": meta["quarter"],
                "year":    int(meta["year"]),
                "content": content,
            })

    if skipped:
        print(f"  Skipped {len(skipped)} file(s) (unrecognised name or empty).")

    return transcripts


def save_transcript(transcript: dict):
    """
    Saves a single transcript dict as JSON to data/raw/transcripts/.
    Filename format matches what the rest of the pipeline expects:
        TICKER_YYYY-MM-DD.json
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    ticker = transcript["symbol"]
    date   = transcript["date"][:10]  # already YYYY-MM-DD
    path   = os.path.join(OUT_DIR, f"{ticker}_{date}.json")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(transcript, f, indent=2)


def main():
    print(f"Loading transcripts from: {KAGGLE_DIR}")
    all_transcripts = load_transcript_files(KAGGLE_DIR)
    print(f"Found {len(all_transcripts)} transcript file(s).")

    # Optionally filter to the tickers and date range set in config.py
    allowed_tickers = {t.upper() for t in TICKERS} if TICKERS else None

    saved = 0
    skipped = 0
    for t in all_transcripts:
        if allowed_tickers and t["symbol"] not in allowed_tickers:
            skipped += 1
            continue
        if not (DATE_START <= t["date"] <= DATE_END):
            skipped += 1
            continue
        save_transcript(t)
        saved += 1

    print(f"Saved {saved} transcript(s) → {OUT_DIR}")
    if skipped:
        print(f"Skipped {skipped} transcript(s) outside configured tickers/date range.")


if __name__ == "__main__":
    main()