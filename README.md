# Time Series Forecasting for Portfolio Management Optimization

## Overview
This repository contains a full workflow for applying time series forecasting and portfolio optimization to historical financial data. The project is designed around the GMF Investments challenge and focuses on using historical price data for TSLA, BND, and SPY to build forecasting models, evaluate portfolio strategies, and backtest results.

## Business Need
GMF Investments wants to use historical financial data to improve portfolio management decisions by combining forecasting insights with modern portfolio theory. The goal is to identify market trends, evaluate forecasting models, and recommend portfolio allocations that balance risk and return.

## Assets Used
- TSLA: High-growth technology stock with elevated volatility and strong upside potential
- BND: Bond ETF providing stability and lower risk
- SPY: S&P 500 ETF offering broad market exposure

## Data Source
Historical market data is collected from Yahoo Finance for the period from January 1, 2015 to June 30, 2026.

## Project Goals
- Extract and clean financial market data
- Perform exploratory data analysis and risk analysis
- Build and compare forecasting models such as ARIMA/SARIMA and LSTM
- Forecast future market trends for Tesla
- Optimize a portfolio using Modern Portfolio Theory
- Backtest the proposed strategy against a benchmark portfolio

## Learning Outcomes
- API usage with yfinance
- Data wrangling with pandas
- Feature engineering such as returns and rolling volatility
- Statistical modeling with ARIMA/SARIMA
- Deep learning with LSTM
- Model evaluation using MAE, RMSE, and MAPE
- Portfolio optimization and efficient frontier analysis
- Strategy backtesting and performance comparison

## Project Structure
```text
portfolio-optimization/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── reports/
├── scripts/
├── src/
├── tests/
├── README.md
├── requirements.txt
└── .gitignore
```

## Tasks

### Task 1 - Preprocess and Explore the Data
Objective: Load, clean, and understand the data before forecasting.

Included work:
- Download historical data for TSLA, BND, and SPY
- Clean missing values and ensure appropriate data types
- Perform EDA with visualizations
- Analyze volatility and risk metrics
- Run stationarity tests such as the Augmented Dickey-Fuller test

### Task 2 - Build Time Series Forecasting Models
Objective: Build and compare forecasting models for Tesla stock prices.

Included work:
- Split the data chronologically into training and test sets
- Fit ARIMA/SARIMA models
- Build and train an LSTM model
- Compare model performance using MAE, RMSE, and MAPE

### Task 3 - Forecast Future Market Trends
Objective: Generate future forecasts and interpret them for investment decisions.

Included work:
- Produce forecasts for the next 6 to 12 months
- Plot predictions with confidence intervals
- Identify trends, opportunities, and risks
- Assess forecast reliability over different horizons

### Task 4 - Optimize Portfolio Based on Forecast
Objective: Build an optimized portfolio using the forecasted return for Tesla and historical returns for the other assets.

Included work:
- Prepare expected returns
- Compute the covariance matrix
- Generate the efficient frontier
- Identify the maximum Sharpe ratio portfolio and minimum volatility portfolio
- Recommend a final portfolio allocation

### Task 5 - Strategy Backtesting
Objective: Validate the strategy against a benchmark portfolio.

Included work:
- Define a backtesting period using the latest available data
- Compare the strategy against a static 60% SPY / 40% BND benchmark
- Simulate portfolio performance and plot cumulative returns
- Evaluate total return, annualized return, Sharpe ratio, and maximum drawdown

## Deliverables
- Jupyter notebook with EDA and visualizations
- Forecasting models and evaluation results
- Future forecast visualization with confidence intervals
- Efficient frontier plot and portfolio recommendation
- Backtesting comparison chart and performance metrics
- Summary report or investment memo

## Submission Timeline
- Interim submission: Sunday, 05 Jul 2026
- Final submission: Tuesday, 07 Jul 2026

## Tech Stack
- Python
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scikit-learn
- tensorflow or keras
- yfinance
- PyPortfolioOpt

## Setup
1. Create and activate a virtual environment
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebooks or scripts in the repository to reproduce the analysis

## Notes
This project follows a practical finance workflow where forecasting models are used as inputs to broader investment decisions rather than as stand-alone price predictors. The emphasis is on evidence-based analysis, risk awareness, and transparent model evaluation.
