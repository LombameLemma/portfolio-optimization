import os
import numpy as np
import pandas as pd
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
# 2. Compute Returns
# -----------------------------
returns = prices.pct_change().dropna()

# Annualized return + risk assumptions
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

# Risk-free rate (approx US treasury baseline)
risk_free_rate = 0.02

# -----------------------------
# 3. Monte Carlo Simulation
# -----------------------------
num_portfolios = 20000
results = np.zeros((4, num_portfolios))

np.random.seed(42)

for i in range(num_portfolios):
    # Random weights
    weights = np.random.random(len(mean_returns))
    weights /= np.sum(weights)

    # Portfolio return
    portfolio_return = np.dot(weights, mean_returns)

    # Portfolio risk
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Sharpe Ratio
    sharpe = (portfolio_return - risk_free_rate) / portfolio_std

    # Store results
    results[0, i] = portfolio_return
    results[1, i] = portfolio_std
    results[2, i] = sharpe
    results[3, i] = i

# -----------------------------
# 4. Best Portfolio (Max Sharpe)
# -----------------------------
max_sharpe_idx = np.argmax(results[2])
best_return = results[0, max_sharpe_idx]
best_risk = results[1, max_sharpe_idx]
best_sharpe = results[2, max_sharpe_idx]

# Reconstruct best weights
best_weights = np.random.random(len(mean_returns))
best_weights /= np.sum(best_weights)

# (Better approach: recompute properly)
def get_best_weights():
    np.random.seed(42)
    best_sharpe_local = -999
    best_w = None

    for _ in range(num_portfolios):
        w = np.random.random(len(mean_returns))
        w /= np.sum(w)

        ret = np.dot(w, mean_returns)
        vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        sharpe = (ret - risk_free_rate) / vol

        if sharpe > best_sharpe_local:
            best_sharpe_local = sharpe
            best_w = w

    return best_w, best_sharpe_local

best_weights, best_sharpe = get_best_weights()

# -----------------------------
# 5. Display Results
# -----------------------------
print("\n📊 Optimal Portfolio (Max Sharpe Ratio)")
print("--------------------------------------")
for i, ticker in enumerate(mean_returns.index):
    print(f"{ticker}: {best_weights[i]:.2%}")

portfolio_return = np.dot(best_weights, mean_returns)
portfolio_vol = np.sqrt(np.dot(best_weights.T, np.dot(cov_matrix, best_weights)))

print("\nExpected Annual Return:", round(portfolio_return * 100, 2), "%")
print("Annual Volatility:", round(portfolio_vol * 100, 2), "%")
print("Sharpe Ratio:", round(best_sharpe, 3))

# -----------------------------
# 6. Visualization
# -----------------------------
labels = mean_returns.index
plt.figure()
plt.bar(labels, best_weights)
plt.title("Optimal Portfolio Weights (Max Sharpe)")
plt.ylabel("Weight")
plt.show()