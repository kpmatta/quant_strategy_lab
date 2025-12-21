#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import sys
from typing import Any, Dict, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

try:
    from mcp.client.sse import sse_client
except ImportError:  # pragma: no cover - optional transport
    sse_client = None


def _get_stdio_params() -> StdioServerParameters:
    command = os.getenv("MCP_SERVER_COMMAND") or sys.executable
    args_env = os.getenv("MCP_SERVER_ARGS")
    if args_env:
        args = shlex.split(args_env)
    else:
        args = ["-m", "mcp_quant.mcp_server"]
    return StdioServerParameters(command=command, args=args)


def _content_to_value(content: Any) -> Any:
    if isinstance(content, list):
        if len(content) == 1:
            return _item_to_value(content[0])
        return [_item_to_value(item) for item in content]
    return _item_to_value(content)


def _item_to_value(item: Any) -> Any:
    if isinstance(item, (dict, list, str, int, float, bool)) or item is None:
        return item
    text = getattr(item, "text", None)
    if text is not None:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    data = getattr(item, "data", None)
    if data is not None:
        return data
    return item


def _parse_prices(text: str) -> List[float]:
    raw = [value for value in text.replace(",", " ").split() if value]
    prices = []
    for value in raw:
        try:
            num = float(value)
        except ValueError:
            continue
        if num > 0:
            prices.append(num)
    if len(prices) < 5:
        raise ValueError("Provide at least 5 positive price values")
    return prices


async def _call_tool(session: ClientSession, name: str, args: Dict[str, Any]) -> Any:
    result = await session.call_tool(name, args)
    if getattr(result, "is_error", False):
        message = _content_to_value(getattr(result, "content", "")) or "MCP tool error"
        raise RuntimeError(str(message))
    return _content_to_value(getattr(result, "content", result))


async def run_agent(args: argparse.Namespace) -> None:
    url = os.getenv("MCP_SERVER_URL")
    if url:
        if sse_client is None:
            raise RuntimeError("MCP_SERVER_URL is set but SSE support is unavailable")
        transport_cm = sse_client(url)
    else:
        transport_cm = stdio_client(_get_stdio_params())

    async with transport_cm as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            if args.command == "list":
                result = await _call_tool(session, "list_strategies", {})
            elif args.command == "sample":
                result = await _call_tool(
                    session,
                    "sample_price_series",
                    {
                        "length": args.length,
                        "start": args.start,
                        "drift": args.drift,
                        "vol": args.vol,
                        "seed": args.seed,
                    },
                )
            elif args.command == "backtest":
                if args.prices:
                    prices = _parse_prices(args.prices)
                else:
                    prices = await _call_tool(
                        session,
                        "sample_price_series",
                        {"length": args.sample_length},
                    )
                params: Dict[str, Any] | None = None
                if args.params:
                    params = json.loads(args.params)
                result = await _call_tool(
                    session,
                    "run_backtest",
                    {
                        "prices": prices,
                        "strategy": args.strategy,
                        "params": params,
                        "start_cash": args.start_cash,
                        "fee_bps": args.fee_bps,
                    },
                )
            else:
                raise RuntimeError(f"Unknown command: {args.command}")

    print(json.dumps(result, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MCP agent client for quant strategies")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List strategies")

    sample = subparsers.add_parser("sample", help="Generate sample prices")
    sample.add_argument("--length", type=int, default=240)
    sample.add_argument("--start", type=float, default=100.0)
    sample.add_argument("--drift", type=float, default=0.0005)
    sample.add_argument("--vol", type=float, default=0.01)
    sample.add_argument("--seed", type=int, default=7)

    backtest = subparsers.add_parser("backtest", help="Run a backtest")
    backtest.add_argument("--strategy", required=True)
    backtest.add_argument("--params", help="JSON string of strategy params")
    backtest.add_argument("--prices", help="Comma or space separated prices")
    backtest.add_argument("--sample-length", type=int, default=240)
    backtest.add_argument("--start-cash", type=float, default=10000.0)
    backtest.add_argument("--fee-bps", type=float, default=1.0)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    asyncio.run(run_agent(args))


if __name__ == "__main__":
    main()
