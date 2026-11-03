"""Read-only MCP server over the official growwapi SDK: get_portfolio, get_quote, get_historical_candles.

Order placement, modification and cancellation are deliberately not implemented.
Run with: python -m mcp_server.groww_server
"""

import re
import sys
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_settings  # noqa: E402
from mcp_server import mock_data  # noqa: E402

MAX_HISTORY_DAYS = 180  # Groww allows at most 180 days of daily candles per request.
SYMBOL_PATTERN = re.compile(r"^[A-Z0-9&\-]{1,20}$")
READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=True)


def normalize_symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper()
    if not SYMBOL_PATTERN.match(cleaned):
        raise ValueError(f"Invalid trading symbol: {symbol!r}")
    return cleaned


def clamp_days(days: int) -> int:
    return max(5, min(int(days), MAX_HISTORY_DAYS))


class MockProvider:
    source = "mock"

    def holdings(self) -> list[dict]:
        return mock_data.mock_holdings()

    def quote(self, symbol: str) -> dict:
        if not mock_data.is_known_symbol(symbol):
            raise ValueError(f"Unknown symbol in mock portfolio: {symbol}")
        return mock_data.mock_quote(symbol)

    def candles(self, symbol: str, days: int) -> list[dict]:
        if not mock_data.is_known_symbol(symbol):
            raise ValueError(f"Unknown symbol in mock portfolio: {symbol}")
        return mock_data.mock_candles(symbol, days)


class GrowwProvider:
    source = "groww"

    def __init__(self, api_key: str, api_secret: str):
        self._api_key = api_key
        self._api_secret = api_secret
        self._client = None
        self._token_date: date | None = None

    def _api(self, force_refresh: bool = False):
        from growwapi import GrowwAPI

        # Groww access tokens expire daily, so fetch a fresh one each day.
        if force_refresh or self._client is None or self._token_date != date.today():
            token = GrowwAPI.get_access_token(api_key=self._api_key, secret=self._api_secret)
            self._client = GrowwAPI(token)
            self._token_date = date.today()
        return self._client

    def _call(self, fn):
        from growwapi.groww.exceptions import GrowwAPIAuthenticationException

        try:
            return fn(self._api())
        except GrowwAPIAuthenticationException:
            return fn(self._api(force_refresh=True))

    def holdings(self) -> list[dict]:
        payload = self._call(lambda api: api.get_holdings_for_user(timeout=15))
        return [
            {
                "symbol": h["trading_symbol"],
                "company_name": h["trading_symbol"],
                "isin": h.get("isin", ""),
                "quantity": float(h["quantity"]),
                "average_price": float(h["average_price"]),
            }
            for h in payload.get("holdings", [])
        ]

    def quote(self, symbol: str) -> dict:
        q = self._call(lambda api: api.get_quote(
            trading_symbol=symbol, exchange=api.EXCHANGE_NSE, segment=api.SEGMENT_CASH, timeout=15
        ))
        last_price = float(q["last_price"])
        day_change = float(q.get("day_change") or 0.0)
        ohlc = q.get("ohlc") or {}
        last_trade_ms = q.get("last_trade_time")
        return {
            "symbol": symbol,
            "last_price": last_price,
            "previous_close": round(last_price - day_change, 2),
            "day_change": day_change,
            "day_change_pct": float(q.get("day_change_perc") or 0.0),
            "open": ohlc.get("open"),
            "high": ohlc.get("high"),
            "low": ohlc.get("low"),
            "volume": q.get("volume"),
            "week_52_high": q.get("week_52_high"),
            "week_52_low": q.get("week_52_low"),
            "as_of": datetime.fromtimestamp(last_trade_ms / 1000).date().isoformat() if last_trade_ms else None,
        }

    def candles(self, symbol: str, days: int) -> list[dict]:
        end = datetime.now()
        start = end - timedelta(days=days)
        payload = self._call(lambda api: api.get_historical_candles(
            exchange=api.EXCHANGE_NSE,
            segment=api.SEGMENT_CASH,
            groww_symbol=f"NSE-{symbol}",
            start_time=start.strftime("%Y-%m-%d %H:%M:%S"),
            end_time=end.strftime("%Y-%m-%d %H:%M:%S"),
            candle_interval=api.CANDLE_INTERVAL_DAY,
            timeout=30,
        ))
        return [
            {
                "date": str(c[0])[:10],
                "open": float(c[1]),
                "high": float(c[2]),
                "low": float(c[3]),
                "close": float(c[4]),
                "volume": int(c[5] or 0),
            }
            for c in payload.get("candles", [])
        ]


@lru_cache(maxsize=1)
def get_provider():
    settings = load_settings()
    if settings.use_mock_groww:
        return MockProvider()
    missing = [n for n in settings.missing_credentials() if n.startswith("GROWW")]
    if missing:
        raise RuntimeError(f"USE_MOCK_GROWW=false but these are missing: {', '.join(missing)}")
    return GrowwProvider(settings.groww_api_key, settings.groww_api_secret)


mcp = FastMCP(
    "groww-readonly",
    instructions="Read-only access to one Groww portfolio. No trading or order tools exist.",
    log_level="WARNING",
)


@mcp.tool(annotations=READ_ONLY)
def get_portfolio() -> dict:
    """List the holdings in the portfolio: symbol, company name, ISIN, quantity and average buy price."""
    provider = get_provider()
    return {"source": provider.source, "holdings": provider.holdings()}


@mcp.tool(annotations=READ_ONLY)
def get_quote(symbol: str) -> dict:
    """Get the latest NSE quote for one trading symbol (for example RELIANCE): last price, day change and range."""
    provider = get_provider()
    return {"source": provider.source, **provider.quote(normalize_symbol(symbol))}


@mcp.tool(annotations=READ_ONLY)
def get_historical_candles(symbol: str, days: int = MAX_HISTORY_DAYS) -> dict:
    """Get daily OHLCV candles for one NSE trading symbol over the last `days` calendar days (5 to 180)."""
    provider = get_provider()
    clean = normalize_symbol(symbol)
    span = clamp_days(days)
    return {"source": provider.source, "symbol": clean, "interval": "1day", "days": span, "candles": provider.candles(clean, span)}


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
