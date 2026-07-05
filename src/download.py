import os
import yfinance as yf

# Create folder if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

# Assets to download
tickers = ["TSLA", "SPY", "BND"]

# Date range
start_date = "2015-01-01"
end_date = "2026-06-30"

# Download and save each asset
for ticker in tickers:
    print(f"Downloading {ticker}...")

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False
    )

    filename = f"data/raw/{ticker}.csv"
    data.to_csv(filename)

    print(f"{ticker} saved to {filename}")

print("All downloads completed!")