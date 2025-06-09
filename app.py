import pandas as pd
import streamlit as st


from src.option import Option
from src.computation import (
    black_scholes,
    annualized_historical_vola
)
from src.data_handler import get_latest_price
from src.least_square_mc import lsm_american
from src.monte_carlo_pricing import mc_pricing
from src.visualization import plot_1d_distribution


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.set_page_config(page_title="Option Pricing Dashboard", layout="wide")

st.sidebar.title("📂 Pages")
PAGE = st.sidebar.radio(
    label="Go to:",
    options=["Home", "Monte Carlo", "ML Predictor", "About"],
    label_visibility="collapsed",
    horizontal=False,
)

if PAGE == "Home":
    st.title("Option Pricing Calculator")

    with st.form("input_form", clear_on_submit=False):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            option_type = st.selectbox("Option type", ["call", "put"], index=0)
            ticker = st.text_input("Ticker symbol", value="AAPL")
            strike = st.number_input("Strike price K", value=100.0, step=1.0, format="%.2f")
            maturity = st.number_input("Maturity (days)", value=252, step=1)
        with col2:
            r = (
                st.number_input("Risk‑free rate r (%, annual)", value=5.0, step=0.1, format="%.2f")
                / 100
            )
            vol_pct = st.number_input(
                "Volatility σ (%, annualised) – leave 0 if unknown",
                min_value=0.0,
                step=0.1,
                format="%.2f",
                value=0.0  # default display
            )

            volatility = None if vol_pct == 0 else vol_pct / 100

            style = st.selectbox("Style", ["European", "American"], index=0)
            n_paths = st.number_input("Monte‑Carlo paths", value=50_000, step=5_000, format="%i")

        submitted = st.form_submit_button("🔍 Calculate")

    if submitted:
        S0, data = get_latest_price(ticker)
        if S0 is None:
            st.error(f"Could not fetch price for ticker ‘{ticker}’. Please check the symbol.")
            st.stop()
        if volatility is None:
            volatility = annualized_historical_vola(data)

        # --- build Option object -------------------------------------------
        option_obj = Option(option_type=option_type,
                            strike_price=strike,
                            maturity=int(maturity),
                            risk_free_rate=r,
                            volatility=volatility,
                            underlying_ticker=ticker)

        st.session_state["option_obj"] = option_obj
        st.session_state["underlying_data"] = data

        # --- pricing --------------------------------------------------------
        bs_price = black_scholes(S0=S0, option=option_obj)

        if style == "European":
            derived_price, profit_vector, price_at_maturity= mc_pricing(option_obj, data, int(n_paths))
            fig1 = plot_1d_distribution(profit_vector, r"Profit $P_T$", highlight_value=0)
            fig2 = plot_1d_distribution(price_at_maturity, r"Asset Price $S_T$", default_color='blue')
            method_label = "Monte‑Carlo (European)"
        else:  # American
            derived_price = lsm_american(
                S0,
                option_obj,
                n_paths=int(n_paths),
                option_type=option_obj.option_type,
            )
            method_label = "Longstaff‑Schwartz LSM (American)"

        # --- display --------------------------------------------------------
        st.subheader(f"Latest close for {ticker.upper()}: **{S0:,.2f}**")
        col_bs, col_mc = st.columns(2)
        col_bs.metric("Black‑Scholes price", f"{bs_price:,.4f}")
        col_mc.metric(f"{method_label} price", f"{derived_price:,.4f}")

        col1, col2 = st.columns(2)

        with col1:
            st.pyplot(fig1)

        with col2:
            st.pyplot(fig2)
# ---------------------------------------------------------------------
# Footer & Roadmap
# ---------------------------------------------------------------------
st.sidebar.divider()
st.sidebar.markdown(
    "**Roadmap**\n\n• Add Greeks page\n• Add historical data & implied vol page\n• Add scenario analysis page"
)