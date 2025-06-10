import os
import sys

"""
Add the parent directory to the system path
to import the setup module
This is necessary for the logger to work correctly
when running the script directly
"""
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from src.computation import implied_volatility, black_scholes, black_scholes_greeks
from src.option import Option


def plot_1d_distribution(data, name: str, highlight_value=None, highlight_color='red', default_color='green'):
    fig, ax = plt.subplots(figsize=(5, 4))
    n, bins, patches = plt.hist(data, bins=50, edgecolor='black', color=default_color)
    ax.set_title(name)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.grid(True)

    if highlight_value is not None:
        # Color each bar according to its bin position
        for i in range(len(bins) - 1):
            if bins[i+1] < highlight_value:
                patches[i].set_facecolor(highlight_color)
            else:
                patches[i].set_facecolor(default_color)

    return fig


def plot_1d_data(data, name: str):
    plt.plot(data)
    plt.title(name)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()


def plot_vol_surface(
    df: pd.DataFrame,
    spot: float,
    risk_free_rate: float,
    iv_func=implied_volatility,
    maturity_col: str = "days_to_maturity",
    strike_col: str = "strike",
    price_col: str = "ask",
    colorscale: str = "Viridis",
) -> None:
    """
    Plot an implied-volatility surface from a tidy DataFrame.

    Parameters
    ----------
    df                : DataFrame with columns 〈maturity_days, strike, price〉.
    spot              : Current spot/underlying price S0.
    risk_free_rate    : Risk-free rate of volatility.
    iv_func           : Callable (price, spot, strike, maturity_days) → IV.
    maturity_col      : Column name for maturities in days.
    strike_col        : Column name for strikes.
    price_col         : Column name for option market prices.
    colorscale        : Plotly colour scale.

    Returns
    -------
    None (shows an interactive Plotly surface).
    """

    if maturity_col not in df.columns:
        df = df.reset_index(names=maturity_col)

    # 1) compute implied vol for every row
    iv_series = df.apply(
        lambda r: iv_func(
            C_market=r[price_col], S0=spot, K=r[strike_col], T=r[maturity_col], r=risk_free_rate
        ),
        axis=1,
    )
    df_iv = df.assign(iv=iv_series)
    # todo: plot moneyness and not just strike
    # 2) reshape to 2-D grid: rows = maturities, cols = strikes
    iv_grid = df_iv.pivot_table(
        index=maturity_col, columns=strike_col, values="iv", aggfunc="mean"
    ).sort_index()

    iv_grid = (
        iv_grid.interpolate("linear", axis=0)  # fill down
        .interpolate("linear", axis=1)  # fill across
        .fillna(method="ffill", axis=0)  # edge extension
        .fillna(method="bfill", axis=0)
        .fillna(method="ffill", axis=1)
        .fillna(method="bfill", axis=1)
    )
    # 3) mesh for Plotly
    maturities = iv_grid.index.to_numpy(dtype=float)      # y-axis (days)
    strikes    = iv_grid.columns.to_numpy(dtype=float)    # x-axis
    K_grid, T_grid = np.meshgrid(strikes, maturities, indexing="xy")

    # 4) interactive surface
    fig = go.Figure(
        data=go.Surface(
            x=K_grid,
            y=T_grid,
            z=iv_grid.values,
            colorscale=colorscale,
            showscale=True,
            contours=dict(
                x=dict(show=True, color="black", width=1),
                y=dict(show=True, color="black", width=1),
                z=dict(show=False),
            ),
        )
    )

    fig.update_layout(
        title="Implied Volatility Surface",
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Maturity (days)",
            zaxis_title="Implied Volatility",
        ),
        width=950,
        height=700,
    )
    fig.show()


def plot_greeks_vs_strike(option: Option=None, S0=None, T=None, r=None, sigma=None, option_type="call", strike_range=None):
    greeks = {'Delta': [], 'Gamma': [], 'Vega': [], 'Theta': [], 'Rho': []}
    strikes = np.linspace(*strike_range, 50)

    for K in strikes:
        option = Option(strike_price=K, maturity=T, risk_free_rate=r, volatility=sigma, option_type=option_type) if option is None else option
        g = black_scholes_greeks(S0, option)
        for greek in greeks:
            greeks[greek].append(g[greek])

    fig, ax = plt.subplots(figsize=(10, 6))
    for greek, values in greeks.items():
        ax.plot(strikes, values, label=greek)
    ax.set_title(f"Greeks vs Strike Price (T={T:.2f} yrs)")
    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Greek Value")
    ax.legend()
    ax.grid(True)

    return fig


def plot_greeks_vs_maturity(option: Option=None, S0=None, K=None, r=None, sigma=None, option_type="call", maturity_range=None):
    greeks = {'Delta': [], 'Gamma': [], 'Vega': [], 'Theta': [], 'Rho': []}
    maturities = np.linspace(*maturity_range, 50)

    for T in maturities:
        option = Option(strike_price=K, maturity=T, risk_free_rate=r, volatility=sigma, option_type=option_type) if option is None else option
        g = black_scholes_greeks(S0, option)
        for greek in greeks:
            greeks[greek].append(g[greek])

    fig, ax = plt.subplots(figsize=(10, 6))
    for greek, values in greeks.items():
        ax.plot(maturities, values, label=greek)
    ax.set_title(f"Greeks vs Time to Maturity (K={K})")
    ax.set_xlabel("Time to Maturity (years)")
    ax.set_ylabel("Greek Value")
    ax.legend()
    ax.grid(True)

    return fig

if __name__ == "__main__":
    pass

