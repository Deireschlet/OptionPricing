
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from src.option import Option  # type: ignore
from src.data_handler import fetch_option_data  # type: ignore
from src.visualization import plot_vol_surface  # type: ignore  (contains your helper)


# ------------------------------------------------------------------
# 0.  Make sure session has an Option (your existing guard stays)
# ------------------------------------------------------------------
if "option_obj" not in st.session_state:
    st.warning("Please first price an option on the Home page – no session data found.")
    st.stop()

option_obj: Option = st.session_state["option_obj"]
spot, strike, maturity, rate, vola, opt_type = option_obj.to_tuple()


@st.cache_data(show_spinner="Fetching option chain …")
def get_chain(ticker: str, opt_type: str) -> pd.DataFrame:
    return fetch_option_data(ticker, opt_type)

full_chain = get_chain(option_obj.underlying_ticker, opt_type)

if full_chain.empty:
    st.error("No option-chain data returned.")
    st.stop()


with st.form("surface_controls"):
    n_steps = st.number_input(
        "Strike steps on each side of spot", 1, 30, value=10
    )
    n_mats  = st.number_input(                # NEW
        "Number of expiries to include", 1, 20, value=5,
        help="Closest N maturities (in calendar days)"
    )
    plot_btn = st.form_submit_button("Plot surface")


if plot_btn:
    # Keep N nearest maturities
    mats = sorted(full_chain.index.unique())[: n_mats]
    chain = full_chain.loc[full_chain.index.isin(mats)]

    # Keep ± n_steps strike levels
    strikes_sorted = np.sort(chain["strike"].unique())
    idx_spot = np.searchsorted(strikes_sorted, spot)
    lower, upper = max(0, idx_spot-n_steps), min(len(strikes_sorted), idx_spot+n_steps+1)
    sel_strikes = strikes_sorted[lower:upper]
    chain = chain.loc[chain["strike"].isin(sel_strikes)]

    # Clean: drop rows with volume == 0 and IV outliers
    chain = chain.query("volume > 0")

    fig = plot_vol_surface(chain, spot, rate)
    st.plotly_chart(fig, use_container_width=True)