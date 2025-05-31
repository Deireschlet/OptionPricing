from setup import logger
from setup.logger import log_call


@log_call(logger)
class Option:
    """
    A class representing an option contract.

    This class encapsulates all parameters related to an option contract,
    including option type, strike price, maturity, and other relevant information.
    """

    def __init__(self, option_type, strike_price, maturity, risk_free_rate, volatility, underlying_ticker=None):
        """
        Initialize an Option object.

        Args:
            option_type (str): Type of option - 'call' or 'put'
            strike_price (float): Strike price of the option
            maturity (int): Time to maturity in days
            risk_free_rate (float): Risk-free interest rate (decimal form)
            volatility (float): Implied or historical volatility (decimal form)
            underlying_ticker (str, optional): Ticker symbol of the underlying asset

        Raises:
            ValueError: If invalid parameters are provided
        """
        if option_type not in ['call', 'put']:
            raise ValueError("Option type must be 'call' or 'put'")
        if strike_price <= 0 or maturity <= 0 or risk_free_rate < 0:
            raise ValueError("Invalid option parameters")
        if volatility and volatility <= 0:
            raise ValueError("Volatility must be a positive number")

        self.option_type = option_type
        self.strike_price = strike_price
        self.maturity = maturity  # in days
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility if volatility else None
        self.underlying_ticker = underlying_ticker

    def __str__(self):
        """String representation of the Option."""
        return f"{self.option_type.capitalize()} option on {self.underlying_ticker or 'unknown'} with strike {self.strike_price}, maturity {self.maturity} days"

    def __repr__(self):
        """Detailed representation of the Option."""
        return f"Option(option_type='{self.option_type}', strike_price={self.strike_price}, maturity={self.maturity}, risk_free_rate={self.risk_free_rate}, volatility={self.volatility}, underlying_ticker='{self.underlying_ticker}')"