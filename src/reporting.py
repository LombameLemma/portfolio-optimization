def performance_stats(returns):
    import numpy as np

    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol
    max_dd = ((1 + returns).cumprod().cummax() - (1 + returns).cumprod()).max()

    return {
        "Return": ann_return,
        "Volatility": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd
    }