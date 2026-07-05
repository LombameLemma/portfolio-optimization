import numpy as np
import matplotlib.pyplot as plt
from preprocessing import load_prices

prices = load_prices()
returns = prices.pct_change().dropna()

mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

num_portfolios = 20000

results = []

for _ in range(num_portfolios):
    weights = np.random.random(len(mean_returns))
    weights /= np.sum(weights)

    port_return = np.dot(weights, mean_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    results.append([port_return, port_vol])

results = np.array(results)

plt.figure()
plt.scatter(results[:,1], results[:,0], alpha=0.4)
plt.title("Efficient Frontier (Monte Carlo)")
plt.xlabel("Volatility (Risk)")
plt.ylabel("Return")
plt.grid()
plt.show()