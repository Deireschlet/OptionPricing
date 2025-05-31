"""Data handler module for fetching and processing stock data"""

import os
import sys

"""
Add the parent directory to the system path
to import the setup module
This is necessary for the logger to work correctly
when running the script directly
"""
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import yfinance as yf
from setup import logger
from setup.logger import log_call


@log_call(logger)
def get_user_input():
    try:
        option_type = input("Enter option type (put or call): ")
        ticker = input("Enter ticker symbol: ")
        strike_price = float(input("Enter strike price: "))
        maturity = int(input("Enter maturity (in days): "))
        risk_free_rate = float(input("Enter risk-free rate: "))
        vola = float(input("Enter annualized volatility: "))
    except ValueError as e:
        logger.exception(e)
        exit(1)
    return option_type, ticker, strike_price, maturity, risk_free_rate, vola


@log_call(logger)
def fetch_stock_data(ticker: str, start_date: str, end_date: str):
    """
    Fetch historical stock data from Yahoo Finance.

    Args:
        ticker (str): The stock ticker symbol.
        start_date (str): The start date for fetching data (YYYY-MM-DD).
        end_date (str): The end date for fetching data (YYYY-MM-DD).

    Returns:
        DataFrame: A DataFrame containing the historical stock data.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if len(data) > 0:
            return data
        else:
            logger.warning("No data available for the specified dates.")
            return None
    except Exception as e:
        logger.exception(e)
        return None

    
def plot_1d_distribution(data, name: str, highlight_value=None, highlight_color='red', default_color='green'):
    n, bins, patches = plt.hist(data, bins=50, edgecolor='black', color=default_color)
    plt.title(name)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)

    if highlight_value is not None:
        # Color each bar according to its bin position
        for i in range(len(bins) - 1):
            if bins[i+1] < highlight_value:
                patches[i].set_facecolor(highlight_color)
            else:
                patches[i].set_facecolor(default_color)

    plt.show()



def plot_1d_data(data, name: str):
    plt.plot(data)
    plt.title(name)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()

if __name__ == "__main__":
    df = fetch_stock_data("AAPL", "2023-01-01", "2023-12-31")

    print(df.head())

