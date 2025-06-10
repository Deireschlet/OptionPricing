import streamlit as st
from src.visualization import plot_greeks_vs_strike, plot_greeks_vs_maturity
from src.option import Option

st.set_page_config(layout="wide")  # better use of screen space
st.title("🏛️ Option Greeks Visualizer")

# Sidebar parameters
st.sidebar.header("Parameters")

option_obj: Option = st.session_state["option_obj"]

spot, strike, maturity, rate, vola, opt_type = option_obj.to_tuple()

S0 = st.sidebar.slider("Spot Price", 50, 150, 100)
sigma = st.sidebar.slider("Volatility", 0.05, 1.0, 0.2)
r = st.sidebar.slider("Risk-Free Rate", 0.0, 0.1, 0.01)
option_type = st.sidebar.selectbox("Option Type", ["call", "put"])

plot_choice = st.sidebar.radio("Plot Type", ["Greeks vs Strike", "Greeks vs Time to Maturity"])

# Main area
if plot_choice == "Greeks vs Strike":
    T = st.sidebar.slider("Time to Maturity (years)", 0.01, 2.0, 1.0)
    strike_range = st.sidebar.slider("Strike Range", 50, 150, (80, 120))

    fig = plot_greeks_vs_strike(S0, T, r, sigma, option_type, strike_range)
    st.pyplot(fig, use_container_width=True)

elif plot_choice == "Greeks vs Time to Maturity":
    K = st.sidebar.slider("Strike Price", 50, 150, 100)
    maturity_range = st.sidebar.slider("Maturity Range (years)", 0.01, 2.0, (0.05, 1.5))

    fig = plot_greeks_vs_maturity(S0, K, r, sigma, option_type, maturity_range)
    st.pyplot(fig, use_container_width=True)

