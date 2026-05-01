"""
src/data/fetch_transcripts.py
OWNER: Person 1 (Data pipeline)

Fetches earnings call transcripts from FinancialModelingPrep API.
Saves one JSON file per call to data/raw/transcripts/.

Usage:
    python src/data/fetch_transcripts.py
"""
import os
import time
import requests
from dotenv import load_dotenv
from src.utils.config import DATE_START, DATE_END, TICKERS, DATA_RAW_DIR

load_dotenv()
API_KEY = os.getenv("FMP_API_KEY")
BASE_URL = "https://financialmodelingprep.com/api/v3"
OUT_DIR  = os.path.join(DATA_RAW_DIR, "transcripts")


def get_sp500_tickers():
    """Fetch current S&P 500 constituents from FMP."""
    url = f"{BASE_URL}/sp500_constituent?apikey={API_KEY}"
    resp = requests.get(url)
    resp.raise_for_status()
    return [row["symbol"] for row in resp.json()]


def fetch_transcripts_for_ticker(ticker: str) -> list[dict]:
    """
    Returns list of transcript dicts with keys:
        symbol, date, quarter, year, content
    """
    url = f"{BASE_URL}/earning_call_transcript/{ticker}?apikey={API_KEY}"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def save_transcript(transcript: dict):
    os.makedirs(OUT_DIR, exist_ok=True)
    ticker = transcript["symbol"]
    date   = transcript["date"][:10]  # YYYY-MM-DD
    fname  = f"{ticker}_{date}.json"
    path   = os.path.join(OUT_DIR, fname)
    if not os.path.exists(path):
        import json
        with open(path, "w") as f:
            json.dump(transcript, f, indent=2)


def main():
    tickers = TICKERS or get_sp500_tickers()
    print(f"Fetching transcripts for {len(tickers)} tickers...")

    for i, ticker in enumerate(tickers):
        try:
            transcripts = fetch_transcripts_for_ticker(ticker)
            for t in transcripts:
                if DATE_START <= t["date"][:10] <= DATE_END:
                    save_transcript(t)
            print(f"[{i+1}/{len(tickers)}] {ticker}: {len(transcripts)} calls")
        except Exception as e:
            print(f"[{i+1}/{len(tickers)}] {ticker}: ERROR — {e}")
        time.sleep(0.2)  # rate limit


if __name__ == "__main__":
    main()
