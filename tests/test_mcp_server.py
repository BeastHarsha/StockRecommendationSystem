import asyncio

import pytest

from agent.guardrails import ALLOWED_TOOLS, GuardedToolbox
from config import load_settings
from mcp_server import groww_server, mock_data
from tests.conftest import AS_OF


@pytest.fixture(autouse=True)
def mock_mode(monkeypatch):
    monkeypatch.setenv("USE_MOCK_GROWW", "true")
    groww_server.get_provider.cache_clear()
    yield
    groww_server.get_provider.cache_clear()


def test_server_exposes_only_read_only_tools():
    tools = asyncio.run(groww_server.mcp.list_tools())
    assert {t.name for t in tools} == ALLOWED_TOOLS
    assert all(t.annotations.readOnlyHint and not t.annotations.destructiveHint for t in tools)


def test_tools_return_mock_data():
    portfolio = groww_server.get_portfolio()
    assert portfolio["source"] == "mock"
    assert {h["symbol"] for h in portfolio["holdings"]} == {"RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"}

    quote = groww_server.get_quote(" infy ")
    assert quote["symbol"] == "INFY" and quote["day_change_pct"] == -6.2

    history = groww_server.get_historical_candles("ITC", days=999)
    assert history["days"] == groww_server.MAX_HISTORY_DAYS
    assert history["candles"][-1]["close"] == 405.0


@pytest.mark.parametrize("bad", ["DROP TABLE", "INFY;rm", "", "A" * 21])
def test_invalid_symbols_rejected(bad):
    with pytest.raises(ValueError):
        groww_server.normalize_symbol(bad)


def test_unknown_mock_symbol_rejected():
    with pytest.raises(ValueError, match="Unknown symbol"):
        groww_server.get_quote("WIPRO")


def test_live_mode_requires_credentials(monkeypatch):
    monkeypatch.setenv("USE_MOCK_GROWW", "false")
    monkeypatch.delenv("GROWW_API_KEY", raising=False)
    monkeypatch.delenv("GROWW_API_SECRET", raising=False)
    monkeypatch.setattr(groww_server, "load_settings", lambda: load_settings(env_file=None))
    groww_server.get_provider.cache_clear()
    with pytest.raises(RuntimeError, match="GROWW_API_KEY"):
        groww_server.get_provider()


def test_mock_history_is_consistent_across_windows():
    long = mock_data.mock_candles("INFY", 180, AS_OF)
    short = mock_data.mock_candles("INFY", 10, AS_OF)
    assert long[-len(short):] == short


def test_mock_scripted_moves():
    assert mock_data.mock_quote("INFY", AS_OF)["day_change_pct"] == -6.2
    itc = mock_data.mock_candles("ITC", 14, AS_OF)
    assert round((itc[-1]["close"] / itc[-6]["close"] - 1) * 100, 1) == -7.5


@pytest.mark.integration
def test_stdio_round_trip_through_adapters():
    from agent.mcp_client import groww_tools

    async def run():
        async with groww_tools() as tools:
            toolbox = GuardedToolbox(tools)
            portfolio = await toolbox.call("get_portfolio")
            quote = await toolbox.call("get_quote", {"symbol": "INFY"})
            return toolbox, portfolio, quote

    toolbox, portfolio, quote = asyncio.run(run())
    assert {t.name for t in toolbox.tools()} == ALLOWED_TOOLS and not toolbox.rejected_at_load
    assert len(portfolio["holdings"]) == 5
    assert quote["symbol"] == "INFY"
