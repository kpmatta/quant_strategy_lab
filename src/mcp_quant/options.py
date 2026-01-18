from __future__ import annotations

import math
from typing import Dict, Literal


OptionType = Literal["call", "put"]


class BinomialTree:
    """
    Binomial tree for pricing American options using the Cox-Ross-Rubinstein model.
    """

    def __init__(
        self,
        stock_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        dividend_yield: float,
        num_steps: int,
        option_type: OptionType,
    ):
        self.S = stock_price
        self.K = strike_price
        self.T = time_to_expiry
        self.r = risk_free_rate
        self.sigma = volatility
        self.q = dividend_yield
        self.N = num_steps
        self.option_type = option_type.lower()

        # Basic validation first (before calculations)
        if self.sigma <= 0:
            raise ValueError(f"Volatility must be positive, got {self.sigma}")
        if self.T <= 0:
            raise ValueError(f"Time to expiry must be positive, got {self.T}")
        if self.N < 1:
            raise ValueError(f"Number of steps must be at least 1, got {self.N}")

        # Calculate tree parameters
        self.dt = self.T / self.N
        self.u = math.exp(self.sigma * math.sqrt(self.dt))
        self.d = 1.0 / self.u
        self.p = (math.exp((self.r - self.q) * self.dt) - self.d) / (self.u - self.d)
        self.discount = math.exp(-self.r * self.dt)

        # Full validation (after parameters are calculated)
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate all input parameters."""
        if self.S <= 0:
            raise ValueError(f"Stock price must be positive, got {self.S}")
        if self.K <= 0:
            raise ValueError(f"Strike price must be positive, got {self.K}")
        if self.T <= 0:
            raise ValueError(f"Time to expiry must be positive, got {self.T}")
        if self.sigma <= 0:
            raise ValueError(f"Volatility must be positive, got {self.sigma}")
        if self.q < 0:
            raise ValueError(f"Dividend yield cannot be negative, got {self.q}")
        if self.N < 1:
            raise ValueError(f"Number of steps must be at least 1, got {self.N}")
        if self.option_type not in ("call", "put"):
            raise ValueError(
                f"Option type must be 'call' or 'put', got {self.option_type}"
            )
        if not (0 < self.p < 1):
            raise ValueError(
                f"Risk-neutral probability must be between 0 and 1, got {self.p}. "
                "Check that risk-free rate and dividend yield are reasonable."
            )

    def _intrinsic_value(self, stock_price: float) -> float:
        """Calculate intrinsic value of the option at a given stock price."""
        if self.option_type == "call":
            return max(stock_price - self.K, 0.0)
        else:  # put
            return max(self.K - stock_price, 0.0)

    def price(self) -> float:
        """
        Price the American option using backward induction through the binomial tree.
        """
        # Initialize asset prices at maturity (time step N)
        stock_prices = [self.S * (self.u ** (self.N - 2 * i)) for i in range(self.N + 1)]

        # Initialize option values at maturity
        option_values = [self._intrinsic_value(price) for price in stock_prices]

        # Backward induction through the tree
        for step in range(self.N - 1, -1, -1):
            for i in range(step + 1):
                # Stock price at this node
                stock_price = self.S * (self.u ** (step - 2 * i))

                # Continuation value (expected discounted value)
                continuation_value = self.discount * (
                    self.p * option_values[i] + (1 - self.p) * option_values[i + 1]
                )

                # Intrinsic value (immediate exercise)
                intrinsic_value = self._intrinsic_value(stock_price)

                # American option: take maximum of continuation and exercise
                option_values[i] = max(continuation_value, intrinsic_value)

        return option_values[0]


def price_american_option(
    stock_price: float,
    strike_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
    num_steps: int = 100,
) -> Dict[str, float]:
    """
    Price an American option using the binomial tree model.

    Args:
        stock_price: Current stock price (S)
        strike_price: Strike price (K)
        time_to_expiry: Time to expiration in years (T)
        risk_free_rate: Risk-free interest rate as decimal (r)
        volatility: Volatility as decimal (sigma)
        option_type: 'call' or 'put'
        dividend_yield: Continuous dividend yield as decimal (q), default 0
        num_steps: Number of time steps in binomial tree (N), default 100

    Returns:
        Dictionary with 'price' and 'intrinsic_value'
    """
    tree = BinomialTree(
        stock_price=stock_price,
        strike_price=strike_price,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility,
        dividend_yield=dividend_yield,
        num_steps=num_steps,
        option_type=option_type,
    )

    option_price = tree.price()
    intrinsic_value = tree._intrinsic_value(stock_price)

    return {
        "price": round(option_price, 4),
        "intrinsic_value": round(intrinsic_value, 4),
    }


def calculate_greeks(
    stock_price: float,
    strike_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
    num_steps: int = 100,
) -> Dict[str, float]:
    """
    Calculate option Greeks using finite difference approximations.

    Args:
        stock_price: Current stock price (S)
        strike_price: Strike price (K)
        time_to_expiry: Time to expiration in years (T)
        risk_free_rate: Risk-free interest rate as decimal (r)
        volatility: Volatility as decimal (sigma)
        option_type: 'call' or 'put'
        dividend_yield: Continuous dividend yield as decimal (q), default 0
        num_steps: Number of time steps in binomial tree (N), default 100

    Returns:
        Dictionary with delta, gamma, theta, vega, and rho
    """
    # Base price
    base_result = price_american_option(
        stock_price, strike_price, time_to_expiry, risk_free_rate,
        volatility, option_type, dividend_yield, num_steps
    )
    base_price = base_result["price"]

    # Delta: dV/dS (1% change in stock price)
    ds = stock_price * 0.01
    price_up = price_american_option(
        stock_price + ds, strike_price, time_to_expiry, risk_free_rate,
        volatility, option_type, dividend_yield, num_steps
    )["price"]
    price_down = price_american_option(
        stock_price - ds, strike_price, time_to_expiry, risk_free_rate,
        volatility, option_type, dividend_yield, num_steps
    )["price"]
    delta = (price_up - price_down) / (2 * ds)

    # Gamma: d²V/dS² (second derivative)
    gamma = (price_up - 2 * base_price + price_down) / (ds ** 2)

    # Theta: -dV/dT (1 day change in time)
    dt = 1.0 / 365.0  # 1 day
    if time_to_expiry > dt:
        price_later = price_american_option(
            stock_price, strike_price, time_to_expiry - dt, risk_free_rate,
            volatility, option_type, dividend_yield, num_steps
        )["price"]
        # Theta is the negative rate of change (time decay)
        theta = -(base_price - price_later) / dt
    else:
        theta = 0.0

    # Vega: dV/dσ (1% change in volatility)
    dvol = 0.01  # 1% absolute change
    price_vol_up = price_american_option(
        stock_price, strike_price, time_to_expiry, risk_free_rate,
        volatility + dvol, option_type, dividend_yield, num_steps
    )["price"]
    vega = price_vol_up - base_price

    # Rho: dV/dr (1% change in risk-free rate)
    dr = 0.01  # 1% absolute change
    price_rate_up = price_american_option(
        stock_price, strike_price, time_to_expiry, risk_free_rate + dr,
        volatility, option_type, dividend_yield, num_steps
    )["price"]
    rho = price_rate_up - base_price

    return {
        "delta": round(delta, 4),
        "gamma": round(gamma, 4),
        "theta": round(theta, 4),
        "vega": round(vega, 4),
        "rho": round(rho, 4),
    }
