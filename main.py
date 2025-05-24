import numpy as np
import pandas as pd
import datetime as dt
from setup import logger, config
from setup.logger import log_call
from src.data_handler import fetch_stock_data, plot_1d_distribution, plot_1d_data
from src.computation import black_scholes, annualized_historical_vola, simulate_stock_price, discounted_avg_payoff \
    , payoff


@log_call(logger)
def main():
    pass

if __name__ == "__main__":
    main()
