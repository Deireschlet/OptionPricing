import datetime as dt
import numpy as np
from setup import logger, config
from setup.logger import log_call
from src.data_handler import get_user_input, fetch_stock_data, fetch_option_data
from src.visualization import plot_vol_surface
from src.monte_carlo_pricing import mc_pricing
from src.least_square_mc import lsm_american
from src.option import Option
from src.computation import black_scholes_greeks, implied_volatility, black_scholes

# todo: create dashboard for data
# todo: pass relevant asset data to vola surface

START_DATE = (dt.date.today() - dt.timedelta(days=365)).strftime("%Y-%m-%d")
END_DATE: str = dt.date.today().strftime("%Y-%m-%d")


@log_call(logger)
def main():
    """option_type, ticker, strike_price, maturity, risk_free_rate, vola = get_user_input()
    option = Option(option_type, strike_price, maturity, risk_free_rate, vola, ticker)

    df = fetch_stock_data(ticker, START_DATE, END_DATE)
    S0 = df['Close'].iloc[-1]
    mc_price = mc_pricing(option, df)
    bs_price = black_scholes(S0, option)
    lsm_price = lsm_american(S0, option, poly_degree=2, q=0.0, option_type=option_type)
    greeks = black_scholes_greeks(S0, option)
    iv = implied_volatility(bs_price, S0, option)"""

    ticker = "AAPL"
    call_data = fetch_option_data(ticker, "call")

    plot_vol_surface(call_data, spot=166.97, risk_free_rate=0.03)

    print("Hallo")


if __name__ == "__main__":
    main()
