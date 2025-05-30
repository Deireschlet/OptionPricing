import numpy as np
import pandas as pd
import datetime as dt
from setup import logger, config
from setup.logger import log_call
from src.data_handler import get_user_input, fetch_stock_data
from src.monte_carlo_pricing import mc_pricing
from src.least_square_mc import lsm_american


@log_call(logger)
def main():
    option_type, ticker, strike_price, maturity, risk_free_rate, vola = get_user_input()
    df = fetch_stock_data(ticker, "2023-01-01", dt.date.today().strftime("%Y-%m-%d"))

    # todo: create data frame to add all return values together

    mc_pricing(option_type, df, strike_price, maturity, risk_free_rate, vola)
    lsm_american(option_type, df, strike_price, maturity, risk_free_rate, vola)

if __name__ == "__main__":
    main()
