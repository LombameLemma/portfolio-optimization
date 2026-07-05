import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Data
# -----------------------------
path = "data/raw"

tsla = pd.read_csv(f"{path}/TSLA.csv", index_col=0, parse_dates=True)
spy = pd.read_csv(f"{path}/SPY.csv", index_col=0, parse_dates=True)
bnd = pd.read_csv(f"{path}/BND.csv", index_col=0, parse_dates=True)

# Keep only Adjusted Close (important for finance work)
prices = pd.DataFrame({
    "TSLA": tsla["Adj Close"],
    "SPY": spy["Adj Close"],
    "BND": bnd["Adj Close"]
})

# -----------------------------
# 2. Handle missing values
# -----------------------------
prices = prices.dropna()

print("Data preview:")
print(prices.head())

# -----------------------------
# 3. Normalize prices (start = 100)
# -----------------------------
normalized = prices / prices.iloc[0] * 100

# -----------------------------
# 4. Plot price trends
# -----------------------------
plt.figure()
normalized.plot(figsize=(12, 6))
plt.title("Normalized Price Performance (Start = 100)")
plt.xlabel("Date")
plt.ylabel("Normalized Price")
plt.grid()
plt.legend()
plt.show()

# -----------------------------
# 5. Calculate daily returns
# -----------------------------
returns = prices.pct_change().dropna()

# -----------------------------
# 6. Plot returns
# -----------------------------
plt.figure()
returns.plot(figsize=(12, 6), alpha=0.7)
plt.title("Daily Returns")
plt.xlabel("Date")
plt.ylabel("Return")
plt.grid()
plt.legend()
plt.show()

# -----------------------------
# 7. Correlation matrix
# -----------------------------
corr = returns.corr()
print("\nCorrelation Matrix:")
print(corr)

# Heatmap
plt.figure()
plt.imshow(corr, cmap="coolwarm", interpolation="none")
plt.colorbar()
plt.xticks(range(len(corr)), corr.columns)
plt.yticks(range(len(corr)), corr.columns)
plt.title("Asset Correlation Heatmap")
plt.show()