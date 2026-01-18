from __future__ import annotations

import unittest

from mcp_quant import options


class OptionsTests(unittest.TestCase):
    """Test suite for binomial tree options pricing."""

    def test_call_price_atm(self) -> None:
        """Test at-the-money call option pricing."""
        result = options.price_american_option(
            stock_price=100.0,
            strike_price=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.20,
            option_type="call",
            dividend_yield=0.0,
            num_steps=100,
        )
        self.assertIn("price", result)
        self.assertIn("intrinsic_value", result)
        # ATM call should have positive price
        self.assertGreater(result["price"], 0)
        # ATM call intrinsic value should be 0
        self.assertEqual(result["intrinsic_value"], 0.0)
        # Verify price is in reasonable range (approximately $10-12 for these params)
        self.assertGreater(result["price"], 8.0)
        self.assertLess(result["price"], 15.0)

    def test_put_price_atm(self) -> None:
        """Test at-the-money put option pricing."""
        result = options.price_american_option(
            stock_price=100.0,
            strike_price=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.20,
            option_type="put",
            dividend_yield=0.0,
            num_steps=100,
        )
        self.assertGreater(result["price"], 0)
        self.assertEqual(result["intrinsic_value"], 0.0)
        # American put should be worth less than call for same params (no dividends)
        # Verify price is in reasonable range (approximately $5-8 for these params)
        self.assertGreater(result["price"], 3.0)
        self.assertLess(result["price"], 10.0)

    def test_call_itm(self) -> None:
        """Test in-the-money call option pricing."""
        result = options.price_american_option(
            stock_price=120.0,
            strike_price=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.20,
            option_type="call",
            num_steps=100,
        )
        # ITM call intrinsic value is S - K
        self.assertEqual(result["intrinsic_value"], 20.0)
        # Option price should be greater than intrinsic value (time value)
        self.assertGreater(result["price"], result["intrinsic_value"])

    def test_put_itm(self) -> None:
        """Test in-the-money put option pricing."""
        result = options.price_american_option(
            stock_price=80.0,
            strike_price=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.20,
            option_type="put",
            num_steps=100,
        )
        # ITM put intrinsic value is K - S
        self.assertEqual(result["intrinsic_value"], 20.0)
        # Option price should be greater than or equal to intrinsic value
        self.assertGreaterEqual(result["price"], result["intrinsic_value"])

    def test_call_otm(self) -> None:
        """Test out-of-the-money call option pricing."""
        result = options.price_american_option(
            stock_price=80.0,
            strike_price=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.20,
            option_type="call",
            num_steps=100,
        )
        # OTM call intrinsic value is 0
        self.assertEqual(result["intrinsic_value"], 0.0)
        # But should have some time value
        self.assertGreater(result["price"], 0)
        # OTM option should be relatively cheap
        self.assertLess(result["price"], 10.0)

    def test_dividend_yield_effect(self) -> None:
        """Test that dividend yield reduces call price."""
        result_no_div = options.price_american_option(
            stock_price=100.0,
            strike_price=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.20,
            option_type="call",
            dividend_yield=0.0,
            num_steps=100,
        )

        result_with_div = options.price_american_option(
            stock_price=100.0,
            strike_price=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.20,
            option_type="call",
            dividend_yield=0.03,  # 3% dividend yield
            num_steps=100,
        )

        # Dividend yield should reduce call price
        self.assertLess(result_with_div["price"], result_no_div["price"])

    def test_greeks_call(self) -> None:
        """Test Greeks calculation for call option."""
        greeks = options.calculate_greeks(
            stock_price=100.0,
            strike_price=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.20,
            option_type="call",
            num_steps=100,
        )

        self.assertIn("delta", greeks)
        self.assertIn("gamma", greeks)
        self.assertIn("theta", greeks)
        self.assertIn("vega", greeks)
        self.assertIn("rho", greeks)

        # ATM call delta should be around 0.5-0.6
        self.assertGreater(greeks["delta"], 0.4)
        self.assertLess(greeks["delta"], 0.7)

        # Gamma should be positive
        self.assertGreater(greeks["gamma"], 0)

        # Theta should be negative (time decay)
        self.assertLess(greeks["theta"], 0)

        # Vega should be positive (higher vol increases call value)
        self.assertGreater(greeks["vega"], 0)

        # Rho should be positive for calls (higher rates increase call value)
        self.assertGreater(greeks["rho"], 0)

    def test_greeks_put(self) -> None:
        """Test Greeks calculation for put option."""
        greeks = options.calculate_greeks(
            stock_price=100.0,
            strike_price=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.20,
            option_type="put",
            num_steps=100,
        )

        # ATM put delta should be around -0.5
        self.assertLess(greeks["delta"], 0)
        self.assertGreater(greeks["delta"], -0.7)

        # Gamma should be positive
        self.assertGreater(greeks["gamma"], 0)

        # Theta should be negative (time decay)
        self.assertLess(greeks["theta"], 0)

        # Vega should be positive
        self.assertGreater(greeks["vega"], 0)

        # Rho should be negative for puts
        self.assertLess(greeks["rho"], 0)

    def test_invalid_inputs(self) -> None:
        """Test that invalid inputs raise ValueError."""
        # Negative stock price
        with self.assertRaises(ValueError):
            options.price_american_option(
                stock_price=-100.0,
                strike_price=100.0,
                time_to_expiry=1.0,
                risk_free_rate=0.05,
                volatility=0.20,
                option_type="call",
            )

        # Negative strike price
        with self.assertRaises(ValueError):
            options.price_american_option(
                stock_price=100.0,
                strike_price=-100.0,
                time_to_expiry=1.0,
                risk_free_rate=0.05,
                volatility=0.20,
                option_type="call",
            )

        # Zero time to expiry
        with self.assertRaises(ValueError):
            options.price_american_option(
                stock_price=100.0,
                strike_price=100.0,
                time_to_expiry=0.0,
                risk_free_rate=0.05,
                volatility=0.20,
                option_type="call",
            )

        # Invalid option type
        with self.assertRaises(ValueError):
            options.price_american_option(
                stock_price=100.0,
                strike_price=100.0,
                time_to_expiry=1.0,
                risk_free_rate=0.05,
                volatility=0.20,
                option_type="invalid",
            )

        # Negative dividend yield
        with self.assertRaises(ValueError):
            options.price_american_option(
                stock_price=100.0,
                strike_price=100.0,
                time_to_expiry=1.0,
                risk_free_rate=0.05,
                volatility=0.20,
                option_type="call",
                dividend_yield=-0.05,
            )

    def test_num_steps_effect(self) -> None:
        """Test that different number of steps produce similar results."""
        result_10 = options.price_american_option(
            stock_price=100.0,
            strike_price=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.20,
            option_type="call",
            num_steps=10,
        )

        result_100 = options.price_american_option(
            stock_price=100.0,
            strike_price=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.20,
            option_type="call",
            num_steps=100,
        )

        # Prices should be similar (within 10% of each other)
        price_diff = abs(result_10["price"] - result_100["price"])
        avg_price = (result_10["price"] + result_100["price"]) / 2
        self.assertLess(price_diff / avg_price, 0.10)

    def test_time_value_decay(self) -> None:
        """Test that option value decreases as expiry approaches."""
        result_1year = options.price_american_option(
            stock_price=100.0,
            strike_price=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.20,
            option_type="call",
            num_steps=100,
        )

        result_1month = options.price_american_option(
            stock_price=100.0,
            strike_price=100.0,
            time_to_expiry=1.0 / 12.0,  # 1 month
            risk_free_rate=0.05,
            volatility=0.20,
            option_type="call",
            num_steps=100,
        )

        # Longer time to expiry should have higher price
        self.assertGreater(result_1year["price"], result_1month["price"])


if __name__ == "__main__":
    unittest.main()
