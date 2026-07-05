from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima


def split_series(series: pd.Series, test_size: float = 0.2):
    n_test = max(30, int(len(series) * test_size))
    train = series.iloc[:-n_test]
    test = series.iloc[-n_test:]
    return train, test


def fit_arima(train: pd.Series, order=None):
    if order is None:
        try:
            model = auto_arima(
                train,
                seasonal=False,
                suppress_warnings=True,
                stepwise=True,
                error_action="ignore",
            )
            order = model.order
        except Exception:
            order = (1, 1, 1)

    arima_model = ARIMA(train, order=order).fit()
    return arima_model, order


def evaluate_forecast(actual: pd.Series, predicted: pd.Series):
    actual = pd.Series(actual).reset_index(drop=True)
    predicted = pd.Series(predicted).reset_index(drop=True)
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def create_lag_features(series: pd.Series, window: int = 60):
    X, y = [], []
    for i in range(window, len(series)):
        X.append(series.iloc[i - window:i].to_numpy())
        y.append(series.iloc[i])
    return np.array(X), np.array(y)


def fit_sequence_model(train: pd.Series, window: int = 60):
    X, y = create_lag_features(train, window=window)
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        random_state=42,
        max_iter=600,
    )
    model.fit(X, y)
    return model, window


def forecast_sequence_model(model, history: pd.Series, steps: int, window: int = 60):
    values = history.iloc[-window:].to_numpy().astype(float)
    preds = []
    for _ in range(steps):
        x = values[-window:].reshape(1, -1)
        pred = float(model.predict(x)[0])
        preds.append(pred)
        values = np.append(values, pred)
    return np.array(preds)


def run_forecasting(prices: pd.DataFrame, ticker: str = "TSLA"):
    series = prices[ticker].astype(float).sort_index()
    train, test = split_series(series)

    arima_model, arima_order = fit_arima(train)
    arima_forecast = arima_model.get_forecast(steps=len(test)).predicted_mean
    arima_metrics = evaluate_forecast(test, arima_forecast)

    seq_model, window = fit_sequence_model(train)
    seq_forecast = forecast_sequence_model(seq_model, train, steps=len(test), window=window)
    seq_metrics = evaluate_forecast(test.values, seq_forecast)

    return {
        "series": series,
        "train": train,
        "test": test,
        "arima_model": arima_model,
        "arima_order": arima_order,
        "arima_forecast": arima_forecast,
        "arima_metrics": arima_metrics,
        "sequence_model": seq_model,
        "sequence_window": window,
        "sequence_forecast": seq_forecast,
        "sequence_metrics": seq_metrics,
    }