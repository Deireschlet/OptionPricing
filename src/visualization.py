import os
import sys

from src.least_square_mc import TRADING_DAYS

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
from typing import Dict, Tuple, Optional


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
    price_col: str = "lastPrice",
    colorscale: str = "Inferno",
) -> Optional[go.Figure]:
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

    # compute implied vol and moneyness for every row
    iv_series = df.apply(
        lambda r: iv_func(
            C_market=r[price_col], S0=spot, K=r[strike_col], T=r[maturity_col], r=risk_free_rate
        ),
        axis=1,
    )

    moneyness_series = df[strike_col] / df[price_col]

    df_iv = df.assign(iv=iv_series,
                      moneyness=moneyness_series)

    # reshape to 2-D grid: rows = maturities, cols = moneyness
    iv_grid = df_iv.pivot_table(
        index=maturity_col, columns="moneyness", values="iv", aggfunc="mean"
    ).sort_index()

    iv_grid = (
        iv_grid.interpolate("linear", axis=0)
        .interpolate("linear", axis=1)
        .ffill(axis=0)
        .bfill(axis=0)
        .ffill(axis=1)
        .bfill(axis=1)
    )
    # mesh for Plotly
    maturities = iv_grid.index.to_numpy(dtype=float)      # y-axis (days)
    strikes = iv_grid.columns.to_numpy(dtype=float)    # x-axis
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
            xaxis_title="Moneyness",
            yaxis_title="Maturity (days)",
            zaxis_title="Implied Volatility",
        ),
        width=950,
        height=700,
    )
    # fig.show()
    return fig


def plot_greeks_vs_strike(
    *,
    S0: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    strike_range: Optional[Tuple[float, float]] = None,
    num_points: int = 50,
) -> plt.Figure:
    """
    Compute and plot Delta, Gamma, Vega, Theta, Rho versus strike price.

    Parameters
    ----------
    S0           : spot price
    T            : time to maturity (years)
    r            : risk-free rate
    sigma        : volatility
    option_type  : 'call' or 'put'
    strike_range : (low, high); if None → [0.5*S0, 1.5*S0]
    num_points    : number of grid points for the x-axis

    Returns
    -------
    matplotlib Figure

    """

    if strike_range is None:
        strike_range = (0.5 * S0, 1.5 * S0)
    strikes = np.linspace(*strike_range, num=num_points)

    greeks: Dict[str, list] = {
        'Delta': [], 'Gamma': [], 'Vega': [], 'Theta': [], 'Rho': []
    }

    for K in strikes:
        opt = Option(
            spot_price=S0,
            option_type=option_type,
            strike_price=K,
            maturity=int(T * TRADING_DAYS),  # days if your class expects days
            risk_free_rate=r,
            volatility=sigma,
        )
        g = black_scholes_greeks(opt)
        for greek in greeks:
            greeks[greek].append(g[greek])

    fig, ax = plt.subplots(figsize=(10, 6))
    for greek, values in greeks.items():
        ax.plot(strikes, values, label=greek)

    ax.set_title(f"Greeks vs Strike (T={T:.2f} yrs, σ={sigma:.2%})")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Greek value")
    ax.legend()
    ax.grid(True)

    return fig


def plot_greeks_vs_maturity(
    *,
    S0: float,
    K: float,
    r: float,
    sigma: float,
    option_type: str = "call",
    maturity_range: Optional[Tuple[float, float]] = None,   # years
    num_points: int = 50,
) -> plt.Figure:
    """
    Compute and plot Delta, Gamma, Vega, Theta, Rho versus time-to-maturity.

    Parameters
    ----------
    S0            : spot price
    K             : strike price
    r             : risk-free rate
    sigma         : volatility
    option_type   : 'call' or 'put'
    maturity_range: (low, high) in **years**; default → (0.01, 2.0)
    num_points    : number of grid points for the x-axis

    Returns
    -------
    matplotlib Figure
    """

    if maturity_range is None:
        maturity_range = (0.01, 2.0)
    maturities = np.linspace(*maturity_range, num=num_points)

    greeks: Dict[str, list] = {
        'Delta': [], 'Gamma': [], 'Vega': [], 'Theta': [], 'Rho': []
    }

    for T_years in maturities:
        opt = Option(
            spot_price=S0,
            option_type=option_type,
            strike_price=K,
            maturity=int(T_years * TRADING_DAYS),  # convert to days if needed
            risk_free_rate=r,
            volatility=sigma,
        )
        g = black_scholes_greeks(opt)
        for greek in greeks:
            greeks[greek].append(g[greek])

    fig, ax = plt.subplots(figsize=(10, 6))
    for greek, values in greeks.items():
        ax.plot(maturities, values, label=greek)

    ax.set_title(f"Greeks vs Time to Maturity (K={K}, σ={sigma:.2%})")
    ax.set_xlabel("Time to Maturity (years)")
    ax.set_ylabel("Greek value")
    ax.legend()
    ax.grid(True)

    return fig


def plot_greeks_vs_strike_from_option(
    option: "Option",
    *,
    strike_range: tuple | None = None,
    num_points: int = 50,
):
    """
    Wrapper that extracts primitives from an Option object and
    calls `plot_greeks_vs_strike(…)`.
    """
    # unpack:  S0, K, T_days, r, sigma, option_type
    S0, _, T_days, r, sigma, option_type = option.to_tuple()

    T_years = T_days / TRADING_DAYS

    return plot_greeks_vs_strike(
        S0=S0,
        T=T_years,
        r=r,
        sigma=sigma,
        option_type=option_type,
        strike_range=strike_range,
        num_points=num_points,
    )


def plot_greeks_vs_maturity_from_option(
    option: "Option",
    *,
    maturity_range: tuple | None = None,  # in years
    num_points: int = 50,
):
    """
    Wrapper that extracts primitives from an Option object and
    calls `plot_greeks_vs_maturity(…)`.
    """
    # unpack primitives
    S0, K, _, r, sigma, option_type = option.to_tuple()

    return plot_greeks_vs_maturity(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        option_type=option_type,
        maturity_range=maturity_range,
        num_points=num_points,
    )
if __name__ == "__main__":
    pass

