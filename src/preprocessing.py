import pandas as pd

def load_prices(path="data/raw"):
    tsla = pd.read_csv(f"{path}/TSLA.csv", index_col=0, parse_dates=True)
    spy = pd.read_csv(f"{path}/SPY.csv", index_col=0, parse_dates=True)
    bnd = pd.read_csv(f"{path}/BND.csv", index_col=0, parse_dates=True)

    prices = pd.DataFrame({
        "TSLA": tsla["Adj Close"],
        "SPY": spy["Adj Close"],
        "BND": bnd["Adj Close"]
    }).dropna()

    return prices