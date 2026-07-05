# Investment Memo

## Objective
This project evaluates whether time-series forecasting and portfolio optimization can improve investment decisions for a diversified portfolio of TSLA, BND, and SPY.

## Methodology
1. Historical price data was loaded for the three assets from January 2015 to June 2026.
2. Daily returns and volatility measures were computed to understand the risk profile of each asset.
3. Forecasting models were trained on Tesla price data and compared using MAE, RMSE, and MAPE.
4. A portfolio was optimized using the forecasted expected return for TSLA and historical return estimates for the other assets.
5. The resulting strategy was backtested against a simple 60% SPY / 40% BND benchmark.

## Findings
- TSLA shows the highest growth potential but also the greatest volatility.
- The optimized portfolio emphasizes broad market exposure while retaining a meaningful allocation to TSLA.
- The forecasting workflow produced measurable error metrics and supported portfolio construction.
- The backtest suggests the strategy delivered a stronger cumulative return than the benchmark in the sample period.

## Recommendation
The portfolio strategy is a reasonable starting point for further refinement, especially if future work includes more robust forecasting, rolling rebalancing, and a longer out-of-sample validation window.
