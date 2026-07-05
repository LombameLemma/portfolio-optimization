import streamlit as st
from preprocessing import load_prices
import numpy as np

st.title("📊 Portfolio Optimizer Dashboard")

prices = load_prices()
returns = prices.pct_change().dropna()

st.line_chart(prices)

st.write("Returns Preview")
st.dataframe(returns.tail())