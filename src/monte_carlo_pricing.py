import pandas as pd
from setup import logger, config
from setup.logger import log_call
from src.computation import annualized_historical_vola, simulate_stock_paths, discounted_avg_payoff, payoff
from src.option import Option

N_PATHS = config.getint("PROJECT", "simulations")

@log_call(logger)
def mc_pricing(option: Option=None,
               df: pd.DataFrame=None,
               n_paths: int=N_PATHS,
               K=None,
               T=None,
               r=None,
               sigma=None,
               opt_type="call",
               ):

    if option is not None:
        S0, K, T, r, sigma, opt_type = option.to_tuple()

    if sigma is None:
        sigma = annualized_historical_vola(df)
    else:
        sigma = float(option.volatility)

    # mu=r otherwise it does not work
    asset_paths = simulate_stock_paths(df['Close'].iloc[-1],
                                     mu=r,
                                     sigma=sigma,
                                     T=T,
                                     N=n_paths
                                     )

    price_at_maturity = asset_paths[-1]

    mc_price = discounted_avg_payoff(price=price_at_maturity,
                                     strike=K,
                                     r=r,
                                     T=T,
                                     option_type=opt_type
                                     )

    payoff_vector = payoff(price=price_at_maturity,
                           strike=K,
                           option_type=opt_type
                           )

    profit_vector = payoff_vector - mc_price

    return mc_price, profit_vector, price_at_maturity

if __name__ == "__main__":
    mc_pricing()