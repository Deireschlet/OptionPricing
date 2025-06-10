import os
import sys
import numpy as np
import pandas as pd
from scipy.constants import sigma
from scipy.stats import norm
from scipy.optimize import brentq
from src.option import Option

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from setup import logger, config
from setup.logger import log_call

TRADING_DAYS = config.getint("PROJECT", "trading_days")
YEAR = config.getfloat("PROJECT", "year")

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
def simulate_stock_paths(S0: float, mu: float=0, sigma: float=None, T: int=None, N: int=None) -> np.ndarray:
    """
    Simulate stock prices using the Geometric Brownian Motion model. Default without drift

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
def black_scholes(option: Option=None, S0=None,  K=None, T=None, r=None, sigma=None, option_type="call"):
    """
    Calculate the price of a European-style option using the Black-Scholes formula.

    Args:
        option (Option): An Option object containing the option parameters.
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
    if option is not None:
        S0, K, T, r, sigma, option_type = option.to_tuple()

    t_years = T / YEAR
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
        return np.mean(np.maximum(price - strike, 0)) * np.exp(-r * T / TRADING_DAYS)
    elif option_type == "put":
        return np.mean(np.maximum(strike - price, 0)) * np.exp(-r * T / TRADING_DAYS)
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


@log_call(logger)
def black_scholes_greeks(option: Option=None, S0=None, K=None, T=None, r=None, sigma=None, option_type="call"):
    """
    Calculates the Black-Scholes Greeks for a European call or put option.

    Parameters:
    S0 : float - current stock price
    option : Option object with strike price, maturity and risk-free rate set
    Returns:
    Dictionary of Greeks: Delta, Gamma, Vega, Theta, Rho
    """

    if option is not None:
        S0, K, T, r, sigma, option_type = option.to_tuple()

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    nd1 = norm.pdf(d1)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    N_neg_d2 = norm.cdf(-d2)

    gamma = nd1 / (S0 * sigma * np.sqrt(T))
    vega = S0 * nd1 * np.sqrt(T) / 100  # per 1% change in vol

    if option_type == 'call':
        delta = Nd1
        theta = (-S0 * nd1 * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * Nd2) / YEAR  # per day
        rho = K * T * np.exp(-r * T) * Nd2 / 100       # per 1% rate change
    elif option_type == 'put':
        delta = Nd1 - 1
        theta = (-S0 * nd1 * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * N_neg_d2) / YEAR
        rho = -K * T * np.exp(-r * T) * N_neg_d2 / 100
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return {
        'Delta': delta,
        'Gamma': gamma,
        'Vega': vega,
        'Theta': theta,
        'Rho': rho
    }


@log_call(logger)
def implied_volatility(C_market, S0, option: Option=None, K=None, T=None, r=None, option_type="call"):
    if option:
        def objective(sigma):
            option.volatility = sigma  # update the volatility
            return black_scholes(S0, option) - C_market
    else:
        def objective(sigma):
            return black_scholes(S0, K=K, T=T, r=r, sigma=sigma, option_type=option_type) - C_market
    try:
        iv = brentq(objective, 1e-6, 5.0)
        return iv
    except ValueError:
        return np.nan


if __name__ == "__main__":
    S0 = 100
    K = 100
    T_days = 252  # 1 year
    r = 0.05
    sigma_annual = 0.2

    # check if put call parity holds to see if black scholes implemented correctly

    call = black_scholes(S0=S0, K=K, T=T_days, r=r, sigma=sigma_annual, option_type="call")
    put = black_scholes(S0=S0, K=K, T=T_days, r=r, sigma=sigma_annual, option_type="put")
    value_to_check = S0 - K * np.exp(-r * (T_days / 252))

    if (call - put) - value_to_check < 1e-6:
        print("Put-call parity holds")
    else:
        print("Put-call parity does not hold")
