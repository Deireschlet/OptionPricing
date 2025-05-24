import numpy as np
import pandas as pd
import datetime as dt
from setup import logger, config
from setup.logger import log_call
from src.data_handler import get_user_input, fetch_stock_data, plot_1d_distribution, plot_1d_data
from src.computation import black_scholes, annualized_historical_vola, simulate_stock_price, discounted_avg_payoff \
    , payoff


@log_call(logger)
def mc_pricing():
    option_type, ticker, strike_price, maturity, risk_free_rate, vola = get_user_input()

    df = fetch_stock_data(ticker, "2023-01-01", dt.date.today().strftime("%Y-%m-%d"))

    if not vola:
        vola = annualized_historical_vola(df)
    else:
        vola = float(vola)

    premium = black_scholes(S0=df['Close'].iloc[-1],
                            K=strike_price,
                            T=maturity,
                            r=risk_free_rate,
                            sigma=vola,
                            option_type=option_type
                            )

    # mu=rate_free_rate otherwise it does not work
    mc_prices = simulate_stock_price(df['Close'].iloc[-1],
                                     mu=risk_free_rate,
                                     sigma=vola,
                                     T=maturity,
                                     N=config.getint("PROJECT", "simulations")
                                     )

    S_T_vector = mc_prices[-1]

    mc_price = discounted_avg_payoff(price=S_T_vector,
                                     strike=strike_price,
                                     r=risk_free_rate,
                                     T=maturity,
                                     option_type=option_type
                                     )

    payoff_vector = payoff(price=S_T_vector,
                           strike=strike_price,
                           option_type=option_type
                           )


    profit_vector = payoff_vector - premium.iloc[0]

    prob_for_profit = np.count_nonzero(profit_vector > 0) / len(profit_vector)

    plot_1d_distribution(profit_vector, r"Profit $P_T$", highlight_value=0)
    plot_1d_distribution(S_T_vector, r"Asset Price $S_T$", default_color='blue')

    print(f"Prob for profit: {prob_for_profit}")
    print(f"MC price: {mc_price}")
    print(f"BS price: {premium}")


if __name__ == "__main__":
    mc_pricing()