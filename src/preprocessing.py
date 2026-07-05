from pathlib import Path
import pandas as pd

TICKERS = ["TSLA", "SPY", "BND"]


def load_prices(raw_dir="data/raw", tickers=None):
    """Load asset price data from the local CSV files."""
    if tickers is None:
        tickers = TICKERS

    frames = {}
    for ticker in tickers:
        path = Path(raw_dir) / f"{ticker}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing data file: {path}")

        df = pd.read_csv(
            path,
            header=None,
            skiprows=[0, 1, 2],
            names=["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"],
        )
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).copy()
        df = df[["Date", "Adj Close", "Close", "High", "Low", "Open", "Volume"]]
        for col in ["Adj Close", "Close", "High", "Low", "Open", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.set_index("Date").sort_index()
        frames[ticker] = df["Adj Close"]

    prices = pd.concat(frames, axis=1).dropna().rename(columns={
        "TSLA": "TSLA",
        "SPY": "SPY",
        "BND": "BND",
    })
    return prices


def compute_returns(prices):
    """Return daily percentage changes for each asset."""
    return prices.pct_change().dropna()