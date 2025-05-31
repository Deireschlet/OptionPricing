import datetime as dt
from setup import logger, config
from setup.logger import log_call
from src.data_handler import get_user_input, fetch_stock_data
from src.monte_carlo_pricing import mc_pricing
from src.least_square_mc import lsm_american
from src.option import Option

# todo: create dashboard for data

START_DATE = (dt.date.today() - dt.timedelta(days=365)).strftime("%Y-%m-%d")
END_DATE: str = dt.date.today().strftime("%Y-%m-%d")


@log_call(logger)
def main():
    option_type, ticker, strike_price, maturity, risk_free_rate, vola = get_user_input()
    option = Option(option_type, strike_price, maturity, risk_free_rate, vola, ticker)

    df = fetch_stock_data(ticker, START_DATE, END_DATE)

    mc_pricing(option, df)
    # lsm_american(option_type, df, strike_price, maturity, risk_free_rate, vola)

if __name__ == "__main__":
    main()
