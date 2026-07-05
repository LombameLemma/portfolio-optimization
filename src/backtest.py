from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def run_backtest(prices: pd.DataFrame, weights: dict, benchmark_weights=None, initial_capital: float = 1.0):
    if benchmark_weights is None:
        benchmark_weights = {"SPY": 0.6, "BND": 0.4}

    monthly_returns = prices.pct_change().dropna().resample("ME").apply(lambda x: (1 + x).prod() - 1)
    monthly_returns = monthly_returns.dropna()

    strategy_values = [initial_capital]
    strategy_returns = []
    for month in monthly_returns.index:
        month_returns = monthly_returns.loc[month]
        port_return = sum(weights.get(asset, 0.0) * month_returns.get(asset, 0.0) for asset in weights)
        strategy_returns.append(port_return)
        strategy_values.append(strategy_values[-1] * (1 + port_return))

    benchmark_values = [initial_capital]
    benchmark_returns = []
    for month in monthly_returns.index:
        month_returns = monthly_returns.loc[month]
        bm_return = sum(benchmark_weights.get(asset, 0.0) * month_returns.get(asset, 0.0) for asset in benchmark_weights)
        benchmark_returns.append(bm_return)
        benchmark_values.append(benchmark_values[-1] * (1 + bm_return))

    strategy_cum = pd.Series(strategy_values[1:], index=monthly_returns.index)
    benchmark_cum = pd.Series(benchmark_values[1:], index=monthly_returns.index)

    def summary(returns_series):
        cum_returns = (1 + returns_series).cumprod() - 1
        ann_return = (1 + cum_returns.iloc[-1]) ** (252 / len(returns_series)) - 1
        sharpe = returns_series.mean() / returns_series.std() * np.sqrt(12)
        peak = (1 + cum_returns).cummax()
        drawdown = ((1 + cum_returns) / peak - 1).min()
        return {
            "total_return": float(cum_returns.iloc[-1]),
            "annualized_return": float(ann_return),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(drawdown),
        }

    strategy_metrics = summary(pd.Series(strategy_returns, index=monthly_returns.index))
    benchmark_metrics = summary(pd.Series(benchmark_returns, index=monthly_returns.index))

    return {
        "strategy_returns": pd.Series(strategy_returns, index=monthly_returns.index),
        "benchmark_returns": pd.Series(benchmark_returns, index=monthly_returns.index),
        "strategy_cumulative": strategy_cum,
        "benchmark_cumulative": benchmark_cum,
        "strategy_metrics": strategy_metrics,
        "benchmark_metrics": benchmark_metrics,
    }


def plot_backtest(result, output_path=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    result["strategy_cumulative"].plot(ax=ax, label="Strategy", linewidth=2)
    result["benchmark_cumulative"].plot(ax=ax, label="60/40 Benchmark", linewidth=2)
    ax.set_title("Strategy vs Benchmark Cumulative Performance")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if output_path:
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
    return fig, ax