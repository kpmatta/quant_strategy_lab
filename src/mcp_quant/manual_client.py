from __future__ import annotations

from typing import Any, Dict, List

from mcp_quant.mcp_client import mcp_client


class ManualModeClient:
    async def list_strategies(self) -> Any:
        return await mcp_client.call_mcp_tool("list_strategies")

    async def sample_price_series(self) -> Any:
        return await mcp_client.call_mcp_tool("sample_price_series", {})

    async def run_backtest(
        self,
        *,
        prices: List[float],
        strategy: str,
        params: Dict[str, float] | None,
        start_cash: float,
        fee_bps: float,
    ) -> Any:
        return await mcp_client.call_mcp_tool(
            "run_backtest",
            {
                "prices": prices,
                "strategy": strategy,
                "params": params,
                "start_cash": start_cash,
                "fee_bps": fee_bps,
            },
        )

    async def price_american_option(
        self,
        *,
        stock_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        option_type: str,
        dividend_yield: float = 0.0,
        num_steps: int = 100,
    ) -> Any:
        return await mcp_client.call_mcp_tool(
            "price_american_option",
            {
                "stock_price": stock_price,
                "strike_price": strike_price,
                "time_to_expiry": time_to_expiry,
                "risk_free_rate": risk_free_rate,
                "volatility": volatility,
                "option_type": option_type,
                "dividend_yield": dividend_yield,
                "num_steps": num_steps,
            },
        )

    async def calculate_option_greeks(
        self,
        *,
        stock_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        option_type: str,
        dividend_yield: float = 0.0,
        num_steps: int = 100,
    ) -> Any:
        return await mcp_client.call_mcp_tool(
            "calculate_option_greeks",
            {
                "stock_price": stock_price,
                "strike_price": strike_price,
                "time_to_expiry": time_to_expiry,
                "risk_free_rate": risk_free_rate,
                "volatility": volatility,
                "option_type": option_type,
                "dividend_yield": dividend_yield,
                "num_steps": num_steps,
            },
        )


manual_client = ManualModeClient()
