"""
src/data/fetch_returns.py
OWNER: Person 1 (Data pipeline)

Downloads post-call stock returns from Yahoo Finance for all tickers
that have transcripts in data/raw/transcripts/.

Usage:
    python src/data/fetch_returns.py
"""
import os
import json
import pandas as pd
import yfinance as yf
from src.utils.config import DATA_RAW_DIR

TRANSCRIPTS_DIR = os.path.join(DATA_RAW_DIR, "transcripts")
OUT_PATH        = os.path.join(DATA_RAW_DIR, "returns.parquet")


def get_call_dates() -> dict[str, list[str]]:
    """Returns {ticker: [date, ...]} from saved transcript files."""
    calls = {}
    for fname in os.listdir(TRANSCRIPTS_DIR):
        if not fname.endswith(".json"):
            continue
        ticker, date = fname.replace(".json", "").rsplit("_", 1)
        calls.setdefault(ticker, []).append(date)
    return calls


def fetch_returns_for_ticker(ticker: str, dates: list[str]) -> pd.DataFrame:
    """
    For each call date, downloads price data and computes:
      - raw_return:      close[t+2] / close[t-1] - 1
      - post_open_return: close[t+2] / open[t+1] - 1  ← the interesting one
      - market_return:   SPY return over same window (for abnormal return calc)

    NOTE: t = call date, t+1 = next market open, t+2 = 2 days after call
    """
    # TODO: implement using yf.Ticker(ticker).history(...)
    # Return a DataFrame with columns:
    #   ticker, call_date, raw_return, post_open_return, market_return, abnormal_return
    raise NotImplementedError("Person 1: implement fetch_returns_for_ticker")


def main():
    calls = get_call_dates()
    all_returns = []

    for ticker, dates in calls.items():
        try:
            df = fetch_returns_for_ticker(ticker, dates)
            all_returns.append(df)
        except Exception as e:
            print(f"{ticker}: ERROR — {e}")

    combined = pd.concat(all_returns, ignore_index=True)
    combined.to_parquet(OUT_PATH, index=False)
    print(f"Saved returns for {len(combined)} calls → {OUT_PATH}")


if __name__ == "__main__":
    main()
