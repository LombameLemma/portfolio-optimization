from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.preprocessing import load_prices, compute_returns
from src.forecasting import run_forecasting
from src.portfolio import optimize_portfolio, plot_frontier
from src.backtest import run_backtest, plot_backtest


def main():
    data_dir = Path("data/raw")
    prices = load_prices(raw_dir=str(data_dir))
    returns = compute_returns(prices)

    print("Loaded price data")
    print(prices.head())

    forecast_result = run_forecasting(prices, ticker="TSLA")
    print("ARIMA order:", forecast_result["arima_order"])
    print("ARIMA metrics:", forecast_result["arima_metrics"])
    print("Sequence metrics:", forecast_result["sequence_metrics"])

    tsla_forecast_return = float(forecast_result["series"].pct_change().dropna().mean() * 252)
    portfolio_result = optimize_portfolio(prices, tsla_expected_return=tsla_forecast_return)
    print("Portfolio weights:")
    print(portfolio_result["weights_df"])
    print("Portfolio performance:", portfolio_result["expected_return"], portfolio_result["volatility"], portfolio_result["sharpe"])

    plot_frontier(portfolio_result, output_path="reports/efficient_frontier.png")
    backtest_result = run_backtest(prices[["TSLA", "SPY", "BND"]], portfolio_result["weights"])
    plot_backtest(backtest_result, output_path="reports/backtest.png")

    print("Strategy metrics:", backtest_result["strategy_metrics"])
    print("Benchmark metrics:", backtest_result["benchmark_metrics"])


if __name__ == "__main__":
    main()
