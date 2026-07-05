# Drawdown
cum = (1 + portfolio_returns).cumprod()
peak = cum.cummax()
drawdown = (cum - peak) / peak

max_drawdown = drawdown.min()

print("Max Drawdown:", round(max_drawdown * 100, 2), "%")