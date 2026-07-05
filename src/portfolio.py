from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import risk_models, expected_returns
from pypfopt.efficient_frontier import EfficientFrontier


def optimize_portfolio(prices: pd.DataFrame, tsla_expected_return: float, risk_free_rate: float = 0.02):
    returns = prices.pct_change().dropna()
    mu = expected_returns.mean_historical_return(prices)
    sigma = risk_models.sample_cov(prices)

    annualized_returns = mu.copy()
    annualized_returns["TSLA"] = tsla_expected_return

    ef = EfficientFrontier(annualized_returns, sigma, weight_bounds=(0, 1))
    ef.max_sharpe(risk_free_rate=risk_free_rate)
    weights = ef.clean_weights()

    port_ret, port_vol, sharpe = ef.portfolio_performance(risk_free_rate=risk_free_rate)
    weights_df = pd.DataFrame([weights]).T.rename(columns={0: "weight"})
    weights_df.index.name = "Asset"

    return {
        "weights": weights,
        "weights_df": weights_df,
        "expected_return": port_ret,
        "volatility": port_vol,
        "sharpe": sharpe,
        "returns": returns,
        "covariance": sigma,
        "efficient_frontier": ef,
    }


def plot_frontier(result, output_path=None):
    returns = result["returns"]
    mu = expected_returns.mean_historical_return(returns.add(1).cumprod())
    sigma = risk_models.sample_cov(returns.add(1).cumprod())

    fig, ax = plt.subplots(figsize=(8, 5))
    n_points = 100
    vol_std = np.sqrt(np.diag(sigma.to_numpy()))
    risk_range = np.linspace(float(vol_std.min()), float(vol_std.max()), n_points)
    expected_returns_values = []
    for _ in risk_range:
        weights = np.repeat(1 / len(returns.columns), len(returns.columns))
        expected_returns_values.append(float(np.dot(weights, mu.to_numpy())))

    ax.plot(risk_range, expected_returns_values, color="steelblue", linewidth=1.5, alpha=0.7)
    ax.scatter(result["volatility"], result["expected_return"], color="red", s=80, label="Max Sharpe")
    ax.set_title("Efficient Frontier")
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Expected Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if output_path:
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
    return fig, ax