
from __future__ import annotations

import streamlit as st
from src.computation import black_scholes
from src.monte_carlo_pricing import mc_pricing
from src.least_square_mc import lsm_american
from src.visualization import plot_profit_distribution, plot_monte_carlo_paths
from src.ui.sidebar import contract_badge

st.set_page_config(page_title="Pricing", layout="wide")
st.title("Option Pricing")

if "option_obj" not in st.session_state:
    st.warning("Please set up an option on the *Home* page first.")
    st.stop()

contract_badge()

option_obj  = st.session_state["option_obj"]
price_series = st.session_state["underlying_data"]

with st.form("pricing_ctrl"):
    style = st.selectbox(
        "Pricing style",
        options=["European", "American"],
        index=0 if option_obj.option_type == "call" else 1
    )
    n_paths = st.number_input(
        "Monte-Carlo paths",
        value=250_000,
        step=10_000,
        format="%i"
    )
    run_btn = st.form_submit_button("Calculate")

if run_btn:
    jump_diff_price = jump_diff_profit_vec = None
    jump_diff_at_T = jump_diff_price_paths = None
    # ---------------------------------------------------------------------

    with st.spinner("Pricing…"):
        bs_price = black_scholes(S0=option_obj.spot_price, option=option_obj)

        if style == "European":
            mc_price, profit_vec, price_at_T, price_paths = mc_pricing(
                option_obj, price_series, int(n_paths)
            )
            jump_diff_price, jump_diff_profit_vec, jump_diff_at_T, jump_diff_price_paths = mc_pricing(
                option_obj,
                price_series,
                int(n_paths),
                mode="jump_diff",
            )
            method_label = "Monte-Carlo"
        else:  # American
            mc_price = lsm_american(option_obj, n_paths=int(n_paths))
            method_label = f"Longstaff-Schwartz LSM ({n_paths:,} paths)"

    # ───────────────────────────── Metrics ──────────────────────────────
    if style == "European":
        col1, col2, col3 = st.columns(3)
        col1.metric("Black-Scholes (closed)", f"{bs_price:,.4f}")
        col2.metric(method_label, f"{mc_price:,.4f}")
        col3.metric("Jump-Diffusion MC", f"{jump_diff_price:,.4f}")
    else:  # American
        col1, col2 = st.columns(2)
        col1.metric("Black-Scholes (closed)", f"{bs_price:,.4f}")
        col2.metric(method_label, f"{mc_price:,.4f}")

    # ───────────────────────── Plots (only for European) ────────────────
    if style == "European":
        f1 = plot_monte_carlo_paths(price_paths, option_obj,
                                    max_paths=int(n_paths * 0.001))
        f2 = plot_profit_distribution(price_at_T, profit_vec, option_obj)

        f3 = plot_monte_carlo_paths(jump_diff_price_paths, option_obj,
                                    max_paths=int(n_paths * 0.008))
        f4 = plot_profit_distribution(jump_diff_at_T, jump_diff_profit_vec, option_obj)

        normal_tab, jump_tab = st.tabs(["Normal", "Jump-Diffusion"])

        with normal_tab:
            c1, c2 = st.columns(2, gap="large")
            c1.pyplot(f1, use_container_width=True)
            c2.pyplot(f2, use_container_width=True)

        with jump_tab:
            c3, c4 = st.columns(2, gap="large")
            c3.pyplot(f3, use_container_width=True)
            c4.pyplot(f4, use_container_width=True)

    # ───────────────────────── Session storage ──────────────────────────
    st.session_state["pricing_result"] = {
        "bs_price": bs_price,
        "mc_price": mc_price,
        "jump_diff_price": jump_diff_price,  # will be None for American
        "method": method_label,
        "n_paths": n_paths,
    }
