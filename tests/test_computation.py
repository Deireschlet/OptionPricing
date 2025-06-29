import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.computation import (
    annualized_historical_vola,
    simulate_stock_paths,
    black_scholes,
    discounted_avg_payoff,
    payoff,
    black_scholes_greeks,
    TRADING_DAYS
)


# Helper functions for tests
def create_test_dataframe(days=100, start_price=100, volatility=0.01, seed=42):
    """Create a test dataframe with stock prices"""
    np.random.seed(seed)
    price = start_price
    prices = [price]

    for _ in range(days - 1):
        change = np.random.normal(0, volatility)
        price = price * (1 + change)
        prices.append(price)

    return pd.DataFrame({
        'Close': prices,
        'Date': pd.date_range(start='2023-01-01', periods=days)
    }).set_index('Date')


def test_put_call_parity():
    """Test the put-call parity relationship in Black-Scholes"""
    S0 = 100
    K = 100
    T_days = 252
    r = 0.05
    sigma = 0.2

    call = black_scholes(S0, K, T_days, r, sigma, option_type="call")
    put = black_scholes(S0, K, T_days, r, sigma, option_type="put")
    value_to_check = S0 - K * np.exp(-r * (T_days / 252))

    assert abs((call - put) - value_to_check) < 1e-6


class TestAnnualizedHistoricalVola:

    def test_correct_calculation(self):
        """Test volatility calculation with known values"""
        # Create a deterministic dataframe with known volatility
        df = create_test_dataframe(days=252, volatility=0.01, seed=42)

        # Calculate volatility
        vola = annualized_historical_vola(df)

        # Check if volatility is close to expected range (this will vary based on the random seed)
        assert 0.1 < vola < 0.3

    def test_empty_dataframe(self):
        """Test with empty dataframe"""
        empty_df = pd.DataFrame({'Close': []})
        vola = annualized_historical_vola(empty_df)
        assert vola is None

    def test_single_value(self):
        """Test with single value (should return None due to no returns)"""
        single_df = pd.DataFrame({'Close': [100]})
        vola = annualized_historical_vola(single_df)
        assert vola is None

    def test_original_df_unchanged(self):
        """Test that original dataframe is not modified"""
        df = create_test_dataframe(days=50)
        original_columns = list(df.columns)
        _ = annualized_historical_vola(df)

        assert list(df.columns) == original_columns
        assert 'log_return' not in df.columns


class TestSimulateStockPaths:

    def test_dimensions(self):
        """Test the dimensions of the output array"""
        S0 = 100
        mu = 0.05
        sigma = 0.2
        T = 252
        N = 1000

        paths = simulate_stock_paths(S0, mu, sigma, T, N)

        assert paths.shape == (T + 1, N)
        assert np.all(paths[0] == S0)  # Check initial price

    def test_random_seed(self):
        """Test that different random seeds produce different paths"""
        S0 = 100
        mu = 0.05
        sigma = 0.2
        T = 10
        N = 10

        np.random.seed(42)
        paths1 = simulate_stock_paths(S0, mu, sigma, T, N)

        np.random.seed(43)
        paths2 = simulate_stock_paths(S0, mu, sigma, T, N)

        # Paths should be different with different seeds
        assert not np.allclose(paths1, paths2)

    def test_zero_volatility(self):
        """Test with zero volatility (deterministic growth)"""
        S0 = 100
        mu = 0.05
        sigma = 0.0
        T = 10
        N = 5

        paths = simulate_stock_paths(S0, mu, sigma, T, N)

        # All paths should be identical with zero volatility
        for i in range(1, N):
            assert np.allclose(paths[:, 0], paths[:, i])

        # Check deterministic growth
        dt = 1 / TRADING_DAYS
        for t in range(1, T + 1):
            expected = S0 * np.exp(mu * dt * t)
            assert np.allclose(paths[t, 0], expected, rtol=1e-10)


class TestBlackScholes:

    def test_call_option(self):
        """Test call option pricing with known values"""
        S0 = 100
        K = 100
        T = 252
        r = 0.05
        sigma = 0.2

        price = black_scholes(S0, K, T, r, sigma, option_type="call")

        # For ATM option with 1 year maturity, price should be around 10-12
        assert 10 < price < 12

    def test_put_option(self):
        """Test put option pricing with known values"""
        S0 = 100
        K = 100
        T = 252
        r = 0.05
        sigma = 0.2

        price = black_scholes(S0, K, T, r, sigma, option_type="put")

        # For ATM option with 1 year maturity, price should be around 5-7 with 5% interest rate
        assert 5 < price < 7

    def test_invalid_option_type(self):
        """Test with invalid option type"""
        price = black_scholes(100, 100, 252, 0.05, 0.2, option_type="invalid")
        assert price is None

    def test_zero_volatility_call(self):
        """Test call option with zero volatility (should equal discounted intrinsic value if in the money)"""
        S0 = 100
        K = 90
        T = 252
        r = 0.05
        sigma = 0.0000001  # Very small but non-zero to avoid division by zero

        price = black_scholes(S0, K, T, r, sigma, option_type="call")
        expected = max(0, S0 - K * np.exp(-r * T / 252))

        assert abs(price - expected) < 0.1

    def test_zero_volatility_put(self):
        """Test put option with zero volatility (should equal discounted intrinsic value if in the money)"""
        S0 = 90
        K = 100
        T = 252
        r = 0.05
        sigma = 0.0000001  # Very small but non-zero to avoid division by zero

        price = black_scholes(S0, K, T, r, sigma, option_type="put")
        expected = max(0, K * np.exp(-r * T / 252) - S0)

        assert abs(price - expected) < 0.1


class TestPayoff:

    def test_call_payoff(self):
        """Test call option payoff calculation"""
        prices = np.array([90, 100, 110])
        strike = 100

        result = payoff(prices, strike, option_type="call")
        expected = np.array([0, 0, 10])

        assert np.array_equal(result, expected)

    def test_put_payoff(self):
        """Test put option payoff calculation"""
        prices = np.array([90, 100, 110])
        strike = 100

        result = payoff(prices, strike, option_type="put")
        expected = np.array([10, 0, 0])

        assert np.array_equal(result, expected)

    def test_invalid_option_type(self):
        """Test with invalid option type"""
        prices = np.array([100])
        strike = 100

        result = payoff(prices, strike, option_type="invalid")
        assert result is None


class TestDiscountedAvgPayoff:

    def test_call_discounted_payoff(self):
        """Test discounted average call option payoff"""
        prices = np.array([90, 100, 110])
        strike = 100
        r = 0.05
        T = 252

        result = discounted_avg_payoff(prices, strike, r, T, option_type="call")
        expected = np.mean(np.maximum(prices - strike, 0)) * np.exp(-r * T / 252)

        assert abs(result - expected) < 1e-10

    def test_put_discounted_payoff(self):
        """Test discounted average put option payoff"""
        prices = np.array([90, 100, 110])
        strike = 100
        r = 0.05
        T = 252

        result = discounted_avg_payoff(prices, strike, r, T, option_type="put")
        expected = np.mean(np.maximum(strike - prices, 0)) * np.exp(-r * T / 252)

        assert abs(result - expected) < 1e-10

    def test_invalid_option_type(self):
        """Test with invalid option type"""
        prices = np.array([100])
        strike = 100

        result = discounted_avg_payoff(prices, strike, 0.05, 252, option_type="invalid")
        assert result is None


class TestBlackScholesGreeks:

    def test_call_option_greeks(self):
        """Test Black-Scholes greeks for call option"""
        S = 100
        K = 100
        T = 1.0  # 1 year
        r = 0.05
        sigma = 0.2

        greeks = black_scholes_greeks(S, K, T, r, sigma, option_type='call')

        # Check that all greeks are present
        assert set(greeks.keys()) == {'Delta', 'Gamma', 'Vega', 'Theta', 'Rho'}

        # Delta of ATM call should be close to 0.5
        assert abs(greeks['Delta'] - 0.5) < 0.15

        # Gamma should be positive
        assert greeks['Gamma'] > 0

        # Vega should be positive
        assert greeks['Vega'] > 0

        # Theta should be negative for call options
        assert greeks['Theta'] < 0

        # Rho should be positive for call options
        assert greeks['Rho'] > 0

    def test_put_option_greeks(self):
        """Test Black-Scholes greeks for put option"""
        S = 100
        K = 100
        T = 1.0  # 1 year
        r = 0.05
        sigma = 0.2

        greeks = black_scholes_greeks(S, K, T, r, sigma, option_type='put')

        # Check that all greeks are present
        assert set(greeks.keys()) == {'Delta', 'Gamma', 'Vega', 'Theta', 'Rho'}

        # Delta of ATM put should be close to -0.5
        assert abs(greeks['Delta'] + 0.5) < 0.15

        # Gamma should be positive and same as call option
        assert greeks['Gamma'] > 0

        # Vega should be positive and same as call option
        assert greeks['Vega'] > 0

        # Theta should be negative for put options
        assert greeks['Theta'] < 0

        # Rho should be negative for put options
        assert greeks['Rho'] < 0

    def test_invalid_option_type(self):
        """Test with invalid option type"""
        with pytest.raises(ValueError, match="option_type must be 'call' or 'put'"):
            black_scholes_greeks(100, 100, 1, 0.05, 0.2, option_type='invalid')

    def test_delta_approaches_limits(self):
        """Test that delta approaches its theoretical limits for deep ITM/OTM options"""
        S = 100
        K = 100
        T = 1.0
        r = 0.05
        sigma = 0.2

        # Deep ITM call (high stock price) should have delta close to 1
        deep_itm_call = black_scholes_greeks(200, K, T, r, sigma, option_type='call')
        assert deep_itm_call['Delta'] > 0.9

        # Deep OTM call (low stock price) should have delta close to 0
        deep_otm_call = black_scholes_greeks(50, K, T, r, sigma, option_type='call')
        assert deep_otm_call['Delta'] < 0.1

        # Deep ITM put (low stock price) should have delta close to -1
        deep_itm_put = black_scholes_greeks(50, K, T, r, sigma, option_type='put')
        assert deep_itm_put['Delta'] < -0.9

        # Deep OTM put (high stock price) should have delta close to 0
        deep_otm_put = black_scholes_greeks(200, K, T, r, sigma, option_type='put')
        assert deep_otm_put['Delta'] > -0.1

