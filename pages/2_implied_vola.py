
from __future__ import annotations

import streamlit as st
import pandas as pd

from src.option import Option  # type: ignore
from src.data_handler import fetch_option_data  # type: ignore
from src.visualization import plot_vol_surface  # type: ignore  (contains your helper)


if "option_obj" not in st.session_state:
    st.warning("Please first price an option on the Home page – no session data found.")
    st.stop()

option_obj: Option = st.session_state["option_obj"]
spot, strike, maturity, rate, vola, opt_type = option_obj.to_tuple()

st.title("📉 Implied Volatility Surface")

with st.expander("Option parameters", expanded=True):
    st.markdown(
        f"**Ticker:** {option_obj.underlying_ticker}  \n"
        f"**Spot (S₀):** {spot:,.2f}  \n"
        f"**Type:** {opt_type}  \n"
        f"**Strike (K):** {strike:,.2f}  \n"
        f"**Maturity (days):** {maturity}  \n"
        f"**Risk‑free rate (r):** {rate:.2%}  \n"
        f"**Model vol (σ):** {vola:.2%}"
    )

@st.cache_data(show_spinner="Fetching option chain …")
def get_chain(ticker: str, opt_type: str) -> pd.DataFrame:
    return fetch_option_data(ticker, opt_type)

chain_df = get_chain(option_obj.underlying_ticker, option_obj.option_type)

if chain_df.empty:
    st.error("No option‑chain data returned.")
    st.stop()


fig = plot_vol_surface(
    df=chain_df,
    spot=spot,
    risk_free_rate=rate,
)

st.plotly_chart(fig, use_container_width=True)
