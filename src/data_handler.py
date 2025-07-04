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

from datetime import date, timedelta
import yfinance as yf
from setup import logger
from setup.logger import log_call
import streamlit as st
import pandas as pd


@st.cache_data(show_spinner="Fetching latest price …")
def get_latest_price(ticker: str) -> tuple[float, pd.DataFrame] | None:
    start = (date.today() - timedelta(days=7)).strftime("%Y-%m-%d")
    end = date.today().strftime("%Y-%m-%d")
    df = fetch_stock_data(ticker, start, end)
    if df is not None and not df.empty:
        return float(df["Close"].iloc[-1].item()), df
    return None


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


@log_call(logger)
def fetch_option_data(ticker: str, opt_type: str) -> pd.DataFrame:
    """
    Fetch option chain data from Yahoo Finance for a given ticker and option type.

    Args:
        ticker (str): The stock ticker symbol.
        opt_type (str): Type of option to fetch ('call' or 'put').

    Returns:
        pd.DataFrame: A DataFrame containing option chain data indexed by days_to_maturity.
    """
    if opt_type not in {"call", "put"}:
        raise ValueError("Invalid option type. Please enter 'call' or 'put'.")

    try:
        data = yf.Ticker(ticker)
        maturities = data.options
        if not maturities:
            raise ValueError(f"No option maturities found for ticker '{ticker}'.")
    except Exception as e:
        logger.error(f"Failed to initialize ticker '{ticker}': {e}")
        return pd.DataFrame()

    today = pd.Timestamp.today().normalize()
    df_list = []

    for date_str in maturities:
        try:
            chain = data.option_chain(date_str)
            df = chain.calls if opt_type == "call" else chain.puts

            if df.empty:
                logger.warning(f"No {opt_type} data for maturity {date_str}. Skipping.")
                continue

            maturity_date = pd.Timestamp(date_str)
            days_to_maturity = (maturity_date - today).days
            df["days_to_maturity"] = days_to_maturity

            df_list.append(df)

        except Exception as e:
            logger.warning(f"Failed to fetch data for {ticker} {opt_type} {date_str}: {e}")
            continue

    if not df_list:
        logger.warning(f"No option data retrieved for {ticker} ({opt_type}).")
        return pd.DataFrame()

    final_df = pd.concat(df_list, ignore_index=True)
    final_df.set_index("days_to_maturity", inplace=True)
    return final_df



if __name__ == "__main__":
    # df = fetch_stock_data("AAPL", "2023-01-01", "2023-12-31")

    calls = fetch_option_data("AAPL", "call")
    print("df.head()")

