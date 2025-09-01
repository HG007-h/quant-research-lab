# scripts/make_returns.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

def main():
    p = argparse.ArgumentParser(description="Build log-returns matrix from Yahoo Finance")
    p.add_argument("--symbols", type=str, required=True,
                   help="Comma-separated tickers, e.g. AAPL,MSFT,GOOG,AMZN,META,NVDA")
    p.add_argument("--start", type=str, default="2018-01-01")
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--out", type=str, default="data/returns.parquet")
    args = p.parse_args()

    tickers = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    print(f"Downloading {len(tickers)} symbols: {tickers}")

    # Download adjusted close prices (auto_adjust=True means 'Close' is already adjusted)
    data = yf.download(
        tickers,
        start=args.start,
        end=args.end,
        progress=False,
        auto_adjust=True,
    )

    if isinstance(data.columns, pd.MultiIndex):
        # multiple tickers -> MultiIndex columns: field x ticker
        df = data["Close"]
    else:
        # single ticker -> flat DataFrame
        df = data[["Close"]].rename(columns={"Close": tickers[0]})

    df = df.sort_index()
    df = df.ffill(limit=3)  # forward-fill up to 3 days
    rets = np.log(df / df.shift(1)).dropna(how="all")
    rets = rets.dropna(axis=1, how="all")  # drop empty columns

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rets.to_parquet(args.out)
    print(f"✅ wrote {args.out} with shape {rets.shape}")

if __name__ == "__main__":
    main()

