import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from setup import logger, config
from setup.logger import log_call

TRADING_DAYS = config.getint("PROJECT", "trading_days")

@log_call(logger)
def annualized_historical_vola(data: pd.DataFrame):
    """
    Calculate the annualized historical volatility of a stock based on its log returns.

    Args:
        data (DataFrame): A DataFrame containing the stock data with log returns.

    Returns:
        float: The historical volatility of the stock.
    """

    data = data.copy()
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1))
    N = len(data['log_return'].dropna())
    if N == 0:
        logger.warning("No log returns available for volatility calculation.")
        return None
    else:
        # for annualized vola multiply by sqrt(252)
        return np.std(data['log_return'].dropna()) * np.sqrt(TRADING_DAYS)


@log_call(logger)
def simulate_stock_price(S0: float, mu: float, sigma: float, T: int, N: int) -> np.ndarray:
    """
    Simulate stock prices using the Geometric Brownian Motion model.

    Args:
        S0 (float): Initial stock price.
        mu (float): Expected return.
        sigma (float): Volatility.
        T (int): Time horizon in days.
        N (int): Number of simulations.

    Returns:
        np.ndarray: Simulated stock prices.
    """
    dt = 1 / TRADING_DAYS # using annualized sigma and mu
    prices = np.zeros((T + 1, N))
    prices[0] = S0
    for i in range(1, T + 1):
        Z = np.random.normal(size=N)
        prices[i] = prices[i - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    return prices

@log_call(logger)
def black_scholes(S0, K, T, r, sigma, option_type="call"):
    """
    Calculate the price of a European-style option using the Black-Scholes formula.

    Args:
        S0 (float): Initial stock price.
        K (float): Strike price.
        T (float): Time to maturity in days.
        r (float): Risk-free interest rate.
        sigma (float): Annualized volatility of the underlying asset.
        option_type (str, optional): Type of option - "call" or "put". Defaults to "call".

    Returns:
        float: The calculated option price.
        None: If option_type is neither "call" nor "put".
    """
    t_years = T / 252
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * t_years) / (sigma * np.sqrt(t_years))
    d2 = d1 - sigma * np.sqrt(t_years)

    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    
    if option_type == "call":
        return S0 * N_d1 - K * np.exp(-r * t_years) * N_d2
    elif option_type == "put":
        return K * np.exp(-r * t_years) * (1 - N_d2) - S0 * (1 - N_d1)
    return None


@log_call(logger)
def discounted_avg_payoff(price: np.ndarray, strike: float, r: float, T: int, option_type: str="call"):
    if option_type == "call":
        return np.mean(np.maximum(price - strike, 0)) * np.exp(-r * T / 252)
    elif option_type == "put":
        return np.mean(np.maximum(strike - price, 0)) * np.exp(-r * T / 252)
    else:
        return None


@log_call(logger)
def payoff(price: np.ndarray, strike: float, option_type: str="call"):
    if option_type == "call":
        return np.maximum(price - strike, 0)
    elif option_type == "put":
        return np.maximum(strike - price, 0)
    else:
        return None


if __name__ == "__main__":
    S0 = 100
    K = 100
    T_days = 252  # 1 year
    r = 0.05
    sigma_annual = 0.2

    # check if put call parity holds to see if black scholes implemented correctly

    call = black_scholes(S0, K, T_days, r, sigma_annual, option_type="call")
    put = black_scholes(S0, K, T_days, r, sigma_annual, option_type="put")
    value_to_check = S0 - K * np.exp(-r * (T_days / 252))

    if (call - put) - value_to_check < 1e-6:
        print("Put-call parity holds")
    else:
        print("Put-call parity does not hold")
