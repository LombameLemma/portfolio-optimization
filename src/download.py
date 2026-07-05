import os
import yfinance as yf

# Create folder if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

tickers = ["TSLA", "SPY", "BND"]

start_date = "2015-01-01"
end_date = "2026-06-30"

for ticker in tickers:
    print(f"Downloading {ticker}...")

    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False   # cleaner output
    )

    # Safety check
    if data.empty:
        print(f"⚠️ No data found for {ticker}")
        continue

    filename = f"data/raw/{ticker}.csv"
    data.to_csv(filename)

    print(f"✅ {ticker} saved to {filename}")

print("All downloads completed!")