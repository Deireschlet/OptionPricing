
from __future__ import annotations

import streamlit as st
from src.computation import black_scholes
from src.monte_carlo_pricing import mc_pricing
from src.least_square_mc import lsm_american
from src.visualization import plot_1d_distribution
from src.ui.sidebar import contract_badge

st.set_page_config(page_title="Pricing", layout="wide")
st.title("💰 Option Pricing")

if "option_obj" not in st.session_state:
    st.warning("Please set up an option on the *Home* page first.")
    st.stop()

contract_badge()

option_obj  = st.session_state["option_obj"]
price_series = st.session_state["underlying_data"]

with st.expander("Current option", expanded=False):
    st.json({
        "ticker"   : option_obj.underlying_ticker,
        "spot"     : f"{option_obj.spot_price:,.2f}",
        "type"     : option_obj.option_type,
        "strike"   : f"{option_obj.strike_price:,.2f}",
        "maturity" : option_obj.maturity,
        "risk_free": f"{option_obj.risk_free_rate:.4f}",
        "vol"      : f"{option_obj.volatility:.4f}"
    })

with st.form("pricing_ctrl"):
    style = st.selectbox(
        "Pricing style",
        options=["European", "American"],
        index=0 if option_obj.option_type == "call" else 1
    )
    n_paths = st.number_input(
        "Monte-Carlo paths",
        value=50_000,
        step=10_000,
        format="%i"
    )
    run_btn = st.form_submit_button("🔍 Calculate")

if run_btn:
    with st.spinner("Pricing…"):
        bs_price = black_scholes(S0=option_obj.spot_price, option=option_obj)

        if style == "European":
            mc_price, profit_vec, price_at_T = mc_pricing(
                option_obj, price_series, int(n_paths)
            )
            method_label = f"Monte-Carlo ({n_paths:,} paths)"
        else:
            mc_price = lsm_american(option_obj, n_paths=int(n_paths))
            method_label = f"Longstaff-Schwartz LSM ({n_paths:,} paths)"

    st.subheader("Results")
    col1, col2 = st.columns(2)
    col1.metric("Black-Scholes", f"{bs_price:,.4f}")
    col2.metric(method_label, f"{mc_price:,.4f}")

    # Optional: MC distribution plots
    if style == "European":
        f1 = plot_1d_distribution(profit_vec, r"Profit $P_T$", highlight_value=0)
        f2 = plot_1d_distribution(price_at_T, r"Asset Price $S_T$", default_color="blue")
        c1, c2 = st.columns(2)
        with c1: st.pyplot(f1)
        with c2: st.pyplot(f2)

    # Store in session for other pages (e.g., Greeks)
    st.session_state["pricing_result"] = {
        "bs_price"   : bs_price,
        "mc_price"   : mc_price,
        "method"     : method_label,
        "n_paths"    : n_paths,
    }
