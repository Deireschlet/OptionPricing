import streamlit as st

from src.visualization import plot_greeks_vs_strike, plot_greeks_vs_maturity
from src.option import Option
from setup import logger, config

TRADING_DAYS = config.getint("PROJECT", "trading_days")
YEAR = config.getint("PROJECT", "year")

if "option_obj" not in st.session_state:
    st.warning("Please first price an option on the Home page – no session data found.")
    st.stop()


st.set_page_config(layout="wide")  # better use of screen space
st.title("🏛️ Option Greeks Visualizer")
plot_choice = st.sidebar.radio("Plot", ["Greeks vs Strike", "Greeks vs Time to Maturity"])

# Sidebar parameters
st.sidebar.header("Parameters")

option_obj: Option = st.session_state["option_obj"]
st.write("option object:", option_obj.to_tuple())
spot, strike, maturity, rate, vola, opt_type = option_obj.to_tuple()

S0 = st.sidebar.slider("Spot Price", spot*0.5, spot*2, spot)
sigma = st.sidebar.slider("Volatility", 0.0, vola*1.5, vola)
r = st.sidebar.slider("Risk-Free Rate", 0.0, 1.0, rate, step=0.01)
option_type = opt_type


# Main area
if plot_choice == "Greeks vs Strike":
    T = st.sidebar.slider("Time to Maturity (years)", 0.0, 2.0, maturity/YEAR, step=0.01)
    strike_range = st.sidebar.slider("Strike Range", strike*0.5, strike*2, (strike, strike*1.2))

    fig = plot_greeks_vs_strike(
        S0=S0,
        T=T,
        r=r,
        sigma=sigma,
        option_type=option_type,
        strike_range=strike_range,
    )
    st.pyplot(fig, use_container_width=True)

elif plot_choice == "Greeks vs Time to Maturity":
    K = st.sidebar.slider("Strike Price", strike*0.5, strike*2, strike)
    maturity_range = st.sidebar.slider("Maturity Range (years)", 0.01, 2.0, (maturity/YEAR, (maturity/YEAR)*1.2))
    fig = plot_greeks_vs_maturity(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        option_type=option_type,
        maturity_range=maturity_range,
    )
    st.pyplot(fig, use_container_width=True)

