"""Deterministic sample portfolio used when USE_MOCK_GROWW=true."""

import zlib
from datetime import date, timedelta

import numpy as np

MOCK_HOLDINGS = [
    {"symbol": "RELIANCE", "company_name": "Reliance Industries", "isin": "INE002A01018", "quantity": 10, "average_price": 2450.0, "end_price": 2890.0},
    {"symbol": "TCS", "company_name": "Tata Consultancy Services", "isin": "INE467B01029", "quantity": 5, "average_price": 3600.0, "end_price": 3420.0},
    {"symbol": "INFY", "company_name": "Infosys", "isin": "INE009A01021", "quantity": 15, "average_price": 1500.0, "end_price": 1610.0},
    {"symbol": "HDFCBANK", "company_name": "HDFC Bank", "isin": "INE040A01034", "quantity": 12, "average_price": 1550.0, "end_price": 1720.0},
    {"symbol": "ITC", "company_name": "ITC", "isin": "INE154A01025", "quantity": 40, "average_price": 430.0, "end_price": 405.0},
]

# Scripted recent moves so the sample portfolio exercises the price-drop alerts.
LAST_DAY_MOVE_PCT = {"RELIANCE": 0.9, "TCS": -1.1, "INFY": -6.2, "HDFCBANK": 0.4}
LAST_WEEK_MOVE_PCT = {"ITC": -7.5}

HISTORY_CALENDAR_DAYS = 400

_BY_SYMBOL = {h["symbol"]: h for h in MOCK_HOLDINGS}


def _last_trading_day(as_of: date) -> date:
    while as_of.weekday() >= 5:
        as_of -= timedelta(days=1)
    return as_of


def _trading_days(end: date, calendar_days: int) -> list[date]:
    start = end - timedelta(days=calendar_days)
    days, current = [], start
    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _closes(symbol: str, count: int) -> list[float]:
    holding = _BY_SYMBOL[symbol]
    rng = np.random.default_rng(zlib.crc32(symbol.encode()))
    returns = list(rng.normal(0.0005, 0.014, size=count - 1))
    if symbol in LAST_WEEK_MOVE_PCT:
        daily = (1 + LAST_WEEK_MOVE_PCT[symbol] / 100) ** (1 / 5) - 1
        returns[-5:] = [daily] * 5
    if symbol in LAST_DAY_MOVE_PCT:
        returns[-1] = LAST_DAY_MOVE_PCT[symbol] / 100

    closes = [holding["end_price"]]
    for r in reversed(returns):
        closes.append(closes[-1] / (1 + r))
    return list(reversed(closes))


def is_known_symbol(symbol: str) -> bool:
    return symbol in _BY_SYMBOL


def mock_holdings() -> list[dict]:
    return [
        {k: h[k] for k in ("symbol", "company_name", "isin", "quantity", "average_price")}
        for h in MOCK_HOLDINGS
    ]


def mock_candles(symbol: str, days: int, as_of: date | None = None) -> list[dict]:
    candles = _full_history(symbol, as_of)
    cutoff = (_last_trading_day(as_of or date.today()) - timedelta(days=days)).isoformat()
    return [c for c in candles if c["date"] >= cutoff]


def _full_history(symbol: str, as_of: date | None) -> list[dict]:
    end = _last_trading_day(as_of or date.today())
    dates = _trading_days(end, HISTORY_CALENDAR_DAYS)
    closes = _closes(symbol, len(dates))
    rng = np.random.default_rng(zlib.crc32(symbol.encode()) + 1)

    candles, prev_close = [], closes[0]
    for day, close in zip(dates, closes):
        open_ = prev_close * (1 + rng.normal(0, 0.004))
        high = max(open_, close) * (1 + abs(rng.normal(0, 0.006)))
        low = min(open_, close) * (1 - abs(rng.normal(0, 0.006)))
        candles.append({
            "date": day.isoformat(),
            "open": round(open_, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "volume": int(rng.integers(500_000, 5_000_000)),
        })
        prev_close = close
    return candles


def mock_quote(symbol: str, as_of: date | None = None) -> dict:
    candles = mock_candles(symbol, 365, as_of)
    last, prev = candles[-1], candles[-2]
    day_change = last["close"] - prev["close"]
    return {
        "symbol": symbol,
        "last_price": last["close"],
        "previous_close": prev["close"],
        "day_change": round(day_change, 2),
        "day_change_pct": round(day_change / prev["close"] * 100, 2),
        "open": last["open"],
        "high": last["high"],
        "low": last["low"],
        "volume": last["volume"],
        "week_52_high": max(c["high"] for c in candles),
        "week_52_low": min(c["low"] for c in candles),
        "as_of": last["date"],
    }
