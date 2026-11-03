from datetime import date

import pytest
from langchain_core.tools import StructuredTool

from mcp_server import mock_data

AS_OF = date(2026, 11, 3)


def make_tools(include_order_tool: bool = False) -> list[StructuredTool]:
    async def get_portfolio() -> dict:
        return {"source": "mock", "holdings": mock_data.mock_holdings()}

    async def get_quote(symbol: str) -> dict:
        return {"source": "mock", **mock_data.mock_quote(symbol.upper(), AS_OF)}

    async def get_historical_candles(symbol: str, days: int = 180) -> dict:
        return {"source": "mock", "symbol": symbol, "interval": "1day", "days": days,
                "candles": mock_data.mock_candles(symbol.upper(), days, AS_OF)}

    async def place_order(symbol: str, quantity: int) -> dict:
        raise AssertionError("place_order must never be called")

    functions = [get_portfolio, get_quote, get_historical_candles]
    if include_order_tool:
        functions.append(place_order)
    return [StructuredTool.from_function(coroutine=f, name=f.__name__, description=f.__name__) for f in functions]


@pytest.fixture
def tools():
    return make_tools()
