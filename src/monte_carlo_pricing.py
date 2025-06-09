import pandas as pd
from setup import logger, config
from setup.logger import log_call
from src.computation import annualized_historical_vola, simulate_stock_paths, discounted_avg_payoff, payoff
from src.option import Option


@log_call(logger)
def mc_pricing(option: Option, df: pd.DataFrame=None, n_paths: int=config.getint("PROJECT", "simulations")):

    if not option.volatility:
        vola = annualized_historical_vola(df)
    else:
        vola = float(option.volatility)

    r = option.risk_free_rate
    T = option.maturity
    K = option.strike_price
    opt_type = option.option_type

    # mu=r otherwise it does not work
    asset_paths = simulate_stock_paths(df['Close'].iloc[-1],
                                     mu=r,
                                     sigma=vola,
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