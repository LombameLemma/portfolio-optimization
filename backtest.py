import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Data
# -----------------------------
path = "data/raw"

tsla = pd.read_csv(f"{path}/TSLA.csv", index_col=0, parse_dates=True)
spy = pd.read_csv(f"{path}/SPY.csv", index_col=0, parse_dates=True)
bnd = pd.read_csv(f"{path}/BND.csv", index_col=0, parse_dates=True)

prices = pd.DataFrame({
    "TSLA": tsla["Adj Close"],
    "SPY": spy["Adj Close"],
    "BND": bnd["Adj Close"]
}).dropna()

# -----------------------------
# 2. Returns
# -----------------------------
returns = prices.pct_change().dropna()

# -----------------------------
# 3. Use SAME weights from portfolio.py
# (replace these with your output weights)
# -----------------------------
weights = np.array([0.45, 0.40, 0.15])  # TSLA, SPY, BND

# -----------------------------
# 4. Portfolio returns
# -----------------------------
portfolio_returns = returns.dot(weights)

# -----------------------------
# 5. Cumulative returns
# -----------------------------
cumulative_portfolio = (1 + portfolio_returns).cumprod()
cumulative_spy = (1 + returns["SPY"]).cumprod()

# -----------------------------
# 6. Plot results
# -----------------------------
plt.figure(figsize=(12, 6))

plt.plot(cumulative_portfolio, label="Optimized Portfolio")
plt.plot(cumulative_spy, label="SPY Benchmark")

plt.title("Portfolio Backtest vs Market")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# 7. Performance metrics
# -----------------------------
annual_return = portfolio_returns.mean() * 252
annual_vol = portfolio_returns.std() * np.sqrt(252)
sharpe = annual_return / annual_vol

print("\n📊 BACKTEST RESULTS")
print("-------------------")
print("Annual Return:", round(annual_return * 100, 2), "%")
print("Annual Volatility:", round(annual_vol * 100, 2), "%")
print("Sharpe Ratio:", round(sharpe, 3))
cum = (1 + portfolio_returns).cumprod()
peak = cum.cummax()
drawdown = (cum - peak) / peak

print("Max Drawdown:", round(drawdown.min() * 100, 2), "%")
annual_return = portfolio_returns.mean() * 252
annual_vol = portfolio_returns.std() * np.sqrt(252)
sharpe = annual_return / annual_vol
plt.figure(figsize=(12,6))
plt.plot((1+portfolio_returns).cumprod(), label="Portfolio")
plt.plot((1+returns["SPY"]).cumprod(), label="SPY")
plt.legend()
plt.title("Portfolio vs Market")
plt.grid()
plt.show()
