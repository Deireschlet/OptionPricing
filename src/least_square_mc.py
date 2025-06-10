import numpy as np
from numpy.polynomial.polynomial import polyval, polyfit
from setup import logger, config
from setup.logger import log_call
from src.computation import simulate_stock_paths, black_scholes
from src.option import Option

TRADING_DAYS = config.getint("PROJECT", "trading_days")


@log_call(logger)
def lsm_american(option: Option,
                 S0: float,
                 K: float,
                 sigma: float,
                 T: int,
                 r: float,
                 n_paths: int = config.getint("PROJECT", "simulations"),
                 poly_degree: int = 2,
                 q:float = 0.0,
                 option_type: str = "put",
                 ) -> float:

    """
    Price an American put or call via the Longstaff-Schwartz LSM algorithm.

    Parameters
    ----------
    S0              : initial stock price
    option          : option object with strike price, maturity and risk-free rate set
    K               : Strike price
    sigma           : Volatility annualized
    T               : Days to maturity
    r               : risk-free interest rate annualized
    n_paths         : number of Monte-Carlo paths
    poly_degree     : polynomial degree for the regression (typically 2)
    q               : continuous dividend yield (annualised)
    option_type     : option type to use (put, call)

    Returns
    -------
    price : float
        LSM estimate of the option value at t = 0, average and already discounted
    """
    if option is not None:
        S0, K, T, r, sigma, option_type = option.to_tuple()

    dt = 1 / TRADING_DAYS
    disc = np.exp(-r * dt)
    paths = simulate_stock_paths(S0, r - q, sigma, T, n_paths)

    if option_type == "put":
        payoffs = np.maximum(K - paths[-1], 0)
    else:
        payoffs = np.maximum(paths[-1] - K, 0)

    # backward induction
    for t in range(T - 1, 0, -1):
        S_t = paths[t]

        if option_type == "put":
            exercise_val = np.maximum(K - S_t, 0)
        else:
            exercise_val = np.maximum(S_t - K, 0)

        itm_mask = exercise_val > 0

        # regression: continuation value estimate
        if np.any(itm_mask):
            X = S_t[itm_mask]
            Y = payoffs[itm_mask] * disc
            coefs = polyfit(X, Y, deg=poly_degree)
            continuation_val = polyval(X, coefs)

            # decide: exercise now?
            exercise_now = exercise_val[itm_mask] > continuation_val

            idx = np.where(itm_mask)[0]
            # overwrite payoffs where early exercise is optimal
            payoffs[idx[exercise_now]] = exercise_val[itm_mask][exercise_now]
            # discount the rest (those not exercised)
            payoffs[idx[~exercise_now]] *= disc

        # paths that were out-of-the-money simply keep their (discounted) future value
        payoffs[~itm_mask] *= disc

    # after looping back to t=1, payoffs are at time 0,
    return payoffs.mean()


if __name__ == "__main__":
    S0     = 100
    K      = 100
    r      = 0.05
    sigma  = 0.2
    T_days = 252

    put_option = Option("put", K, T_days, r, sigma)
    call_option = Option("call", K, T_days, r, sigma)

    # American put via LSM
    am_price = lsm_american(S0, put_option, n_paths=100_000)
    print("LSM American put:", round(am_price, 4))

    # European put (Black-Scholes) for reference
    bs_price = black_scholes(S0, K, T_days, r, sigma, option_type="put")
    print(f"BS put price: {bs_price}")

    # American call via LSM
    am_call_price = lsm_american(S0, call_option, n_paths=100_000, option_type="call")
    print(f"LSM American call: {am_call_price}")

    # European call (Black-Scholes) for reference
    bs_price_call = black_scholes(S0, K, T_days, r, sigma, option_type="call")
    print(f"BS call price: {bs_price_call}")