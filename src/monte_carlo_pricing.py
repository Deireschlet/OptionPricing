import numpy as np
import pandas as pd
from setup import logger, config
from setup.logger import log_call
from src.data_handler import plot_1d_distribution
from src.computation import annualized_historical_vola, simulate_stock_paths, discounted_avg_payoff \
    , payoff
from src.option import Option


@log_call(logger)
def mc_pricing(option: Option, df: pd.DataFrame=None):

    if not option.volatility:
        vola = annualized_historical_vola(df)
    else:
        vola = float(option.volatility)

    r = option.risk_free_rate
    T = option.maturity
    K = option.strike_price
    type = option.option_type

    # mu=r otherwise it does not work
    asset_paths = simulate_stock_paths(df['Close'].iloc[-1],
                                     mu=r,
                                     sigma=vola,
                                     T=T,
                                     N=config.getint("PROJECT", "simulations")
                                     )

    price_at_maturity = asset_paths[-1]

    mc_price = discounted_avg_payoff(price=price_at_maturity,
                                     strike=K,
                                     r=r,
                                     T=T,
                                     option_type=type
                                     )

    payoff_vector = payoff(price=price_at_maturity,
                           strike=K,
                           option_type=type
                           )


    profit_vector = payoff_vector - mc_price

    prob_for_profit = np.count_nonzero(profit_vector > 0) / len(profit_vector)

    plot_1d_distribution(profit_vector, r"Profit $P_T$", highlight_value=0)
    plot_1d_distribution(price_at_maturity, r"Asset Price $S_T$", default_color='blue')

    print(f"Prob for profit: {prob_for_profit}")
    print(f"MC price: {mc_price}")


if __name__ == "__main__":
    mc_pricing()