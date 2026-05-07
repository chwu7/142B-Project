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
from datetime import datetime, timedelta

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
      - raw_return:       close[t+2] / close[t-1] - 1
      - post_open_return: close[t+2] / open[t+1] - 1  <- the interesting one
      - market_return:    SPY return over same window (for abnormal return calc)
      - abnormal_return:  post_open_return - market_return
 
    NOTE: t = call date, t+1 = next market open, t+2 = 2 days after call
    """
    records = []
 
    for call_date in dates:
        t = datetime.strptime(call_date, "%Y-%m-%d")
 
        # Download a window of prices around the call date:
        # start a few days before t (to get close[t-1])
        # end a week after t (to guarantee we capture t+1 and t+2 trading days)
        fetch_start = (t - timedelta(days=5)).strftime("%Y-%m-%d")
        fetch_end   = (t + timedelta(days=10)).strftime("%Y-%m-%d")
 
        try:
            stock_hist = yf.Ticker(ticker).history(start=fetch_start, end=fetch_end)
            spy_hist   = yf.Ticker("SPY").history(start=fetch_start, end=fetch_end)
        except Exception as e:
            print(f"  {ticker} {call_date}: yfinance error — {e}")
            continue
 
        if stock_hist.empty or spy_hist.empty:
            print(f"  {ticker} {call_date}: no price data returned, skipping.")
            continue
 
        # Normalise index to date only (drop timezone)
        stock_hist.index = pd.to_datetime(stock_hist.index).normalize().tz_localize(None)
        spy_hist.index   = pd.to_datetime(spy_hist.index).normalize().tz_localize(None)
 
        # Find the trading days on or after t (t+1, t+2)
        # and the trading day just before t (t-1)
        trading_days     = stock_hist.index.sort_values()
        call_ts          = pd.Timestamp(t)
        days_on_or_after = trading_days[trading_days >= call_ts]
        days_before      = trading_days[trading_days <  call_ts]
 
        if len(days_on_or_after) < 2 or len(days_before) < 1:
            print(f"  {ticker} {call_date}: not enough trading days around call, skipping.")
            continue
 
        day_t_minus_1 = days_before[-1]        # last close before call
        day_t_plus_1  = days_on_or_after[0]    # next open after call
        day_t_plus_2  = days_on_or_after[1]    # close 2 days after call
 
        try:
            close_t_minus_1    = stock_hist.loc[day_t_minus_1, "Close"]
            open_t_plus_1      = stock_hist.loc[day_t_plus_1,  "Open"]
            close_t_plus_2     = stock_hist.loc[day_t_plus_2,  "Close"]
            spy_open_t_plus_1  = spy_hist.loc[day_t_plus_1,    "Open"]
            spy_close_t_plus_2 = spy_hist.loc[day_t_plus_2,    "Close"]
        except KeyError as e:
            print(f"  {ticker} {call_date}: missing price column {e}, skipping.")
            continue
 
        raw_return       = close_t_plus_2 / close_t_minus_1   - 1
        post_open_return = close_t_plus_2 / open_t_plus_1     - 1
        market_return    = spy_close_t_plus_2 / spy_open_t_plus_1 - 1
        abnormal_return  = post_open_return - market_return
 
        records.append({
            "ticker":           ticker,
            "call_date":        call_date,
            "raw_return":       raw_return,
            "post_open_return": post_open_return,
            "market_return":    market_return,
            "abnormal_return":  abnormal_return,
        })
 
    if not records:
        # Return empty DataFrame with correct columns so concat doesn't break
        return pd.DataFrame(columns=[
            "ticker", "call_date", "raw_return",
            "post_open_return", "market_return", "abnormal_return",
        ])
 
    return pd.DataFrame(records)
    #raise NotImplementedError("Person 1: implement fetch_returns_for_ticker")


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
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    combined.to_parquet(OUT_PATH, index=False)
    print(f"Saved returns for {len(combined)} calls -> {OUT_PATH}")


if __name__ == "__main__":
    main()
