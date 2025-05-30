import numpy as np
from numpy.polynomial.polynomial import polyval, polyfit
from setup import logger, config
from setup.logger import log_call
from src.computation import simulate_stock_paths, black_scholes

TRADING_DAYS = config.getint("PROJECT", "trading_days")


@log_call(logger)
def lsm_american(S0: float,
                     K: float,
                     r: float,
                     sigma: float,
                     T_days: int,
                     n_paths: int = 100_000,
                     poly_degree: int = 2,
                     q:float = 0.0,
                     option_type: str = "put",
                     ) -> float:

    """
    Price an American put or call via the Longstaff-Schwartz LSM algorithm.

    Parameters
    ----------
    S0, K, r, sigma : usual Black-Scholes inputs (r, sigma annualised)
    T_days          : maturity in trading days
    n_paths         : number of Monte-Carlo paths
    poly_degree     : polynomial degree for the regression (typically 2)
    q               : continuous dividend yield (annualised)
    option_type     : option type to use (put, call)

    Returns
    -------
    price : float
        LSM estimate of the option value at t = 0, average and already discounted
    """

    dt = 1 / TRADING_DAYS
    disc = np.exp(-r * dt)
    paths = simulate_stock_paths(S0, r - q, sigma, T_days, n_paths)

    if option_type == "put":
        payoffs = np.maximum(K - paths[-1], 0)
    else:
        payoffs = np.maximum(paths[-1] - K, 0)

    # backward induction
    for t in range(T_days - 1, 0, -1):
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

    # American put via LSM
    am_price = lsm_american(S0, K, r, sigma, T_days, n_paths=100_000)
    print("LSM American put:", round(am_price, 4))

    # European put (Black-Scholes) for reference
    bs_price = black_scholes(S0, K, T_days, r, sigma, option_type="put")
    print(f"BS put price: {bs_price}")

    # American call via LSM
    am_call_price = lsm_american(S0, K, r, sigma, T_days, n_paths=100_000, option_type="call")
    print(f"LSM American call: {am_call_price}")

    # European call (Black-Scholes) for reference
    bs_price_call = black_scholes(S0, K, T_days, r, sigma, option_type="call")
    print(f"BS call price: {bs_price_call}")