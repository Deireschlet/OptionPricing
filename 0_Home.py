
from __future__ import annotations

import streamlit as st

from src.option import Option
from src.data_handler import get_latest_price          # returns spot, df
from src.computation import annualized_historical_vola

st.set_page_config(
    page_title="Option Tool",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("Option Contract Setup")

with st.form("setup_form", clear_on_submit=False):
    c1, c2 = st.columns(2, gap="large")

    with c1:
        opt_type  = st.selectbox("Option type", ["call", "put"], index=0)
        ticker    = st.text_input("Ticker", value="AAPL").strip().upper()
        strike    = st.number_input("Strike K", value=100.0, step=1.0, format="%.2f")
        maturity  = st.number_input("Maturity (days)", value=30, step=1, min_value=1)

    with c2:
        r_pct  = st.number_input("Risk-free rate r (%, annual)", value=5.0, step=0.1, format="%.2f")
        vol_pct = st.number_input(
            "Volatility σ (%, annualised) — 0 to auto-estimate",
            min_value=0.0, value=0.0, step=0.1, format="%.2f"
        )
        style = st.selectbox("Style", ["European", "American"], index=0)

    submitted = st.form_submit_button("Save contract")

if submitted:
    spot, price_df = get_latest_price(ticker)

    if spot is None:
        st.error(f"Could not fetch data for {ticker}.")
        st.stop()

    sigma = (vol_pct / 100) if vol_pct else annualized_historical_vola(price_df)

    option_obj = Option(
        spot_price=spot,
        option_type=opt_type,
        strike_price=float(strike),
        maturity=int(maturity),
        risk_free_rate=r_pct/100,
        volatility=sigma,
        underlying_ticker=ticker
    )

    st.session_state["option_obj"]      = option_obj
    st.session_state["underlying_data"] = price_df

    st.success("Contract saved to session.")
    with st.expander("Details", expanded=False):
        st.json({
            "ticker"   : ticker,
            "spot"     : f"{spot:,.2f}",
            "type"     : opt_type,
            "strike"   : f"{strike:,.2f}",
            "maturity_days": maturity,
            "style"    : style,
            "r"        : f"{r_pct:.2f} %",
            "σ"        : f"{sigma:.2%}"
        })

    st.page_link("pages/1_Pricing.py", label="→ Continue to Pricing page")
