"""Deterministic portfolio math. Numbers are computed here, never by the LLM."""

import math


def _r(value: float) -> float:
    return round(value, 2)


def build_holdings(holdings: list[dict], quotes: dict[str, dict]) -> list[dict]:
    rows = []
    for h in holdings:
        quote = quotes.get(h["symbol"])
        qty, avg = float(h["quantity"]), float(h["average_price"])
        invested = qty * avg
        last = float(quote["last_price"]) if quote and quote.get("last_price") is not None else None
        current = qty * last if last is not None else None
        rows.append({
            "symbol": h["symbol"],
            "company_name": h.get("company_name") or h["symbol"],
            "quantity": qty,
            "average_price": _r(avg),
            "last_price": _r(last) if last is not None else None,
            "day_change_pct": _r(float(quote["day_change_pct"])) if quote and quote.get("day_change_pct") is not None else None,
            "invested": _r(invested),
            "current_value": _r(current) if current is not None else None,
            "pnl": _r(current - invested) if current is not None else None,
            "pnl_pct": _r((current - invested) / invested * 100) if current is not None and invested else (0.0 if current is not None else None),
        })
    return rows


def portfolio_totals(rows: list[dict]) -> dict:
    invested = sum(r["invested"] for r in rows)
    if any(r["current_value"] is None for r in rows):
        return {"invested": _r(invested), "current_value": None, "pnl": None, "pnl_pct": None}
    current = sum(r["current_value"] for r in rows)
    return {
        "invested": _r(invested),
        "current_value": _r(current),
        "pnl": _r(current - invested),
        "pnl_pct": _r((current - invested) / invested * 100) if invested else 0.0,
    }


def compute_metrics(symbol: str, candles: list[dict]) -> dict | None:
    if len(candles) < 2:
        return None
    closes = [float(c["close"]) for c in candles]

    peak, max_drawdown = closes[0], 0.0
    for close in closes:
        peak = max(peak, close)
        max_drawdown = min(max_drawdown, (close - peak) / peak)

    returns = [b / a - 1 for a, b in zip(closes, closes[1:])]
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / max(len(returns) - 1, 1)

    return {
        "symbol": symbol,
        "start_date": candles[0]["date"],
        "end_date": candles[-1]["date"],
        "start_close": _r(closes[0]),
        "end_close": _r(closes[-1]),
        "period_high": _r(max(closes)),
        "period_low": _r(min(closes)),
        "return_pct": _r((closes[-1] / closes[0] - 1) * 100),
        "max_drawdown_pct": _r(max_drawdown * 100),
        "volatility_pct": _r(math.sqrt(variance) * math.sqrt(252) * 100),
        "trading_days": len(candles),
    }


def detect_price_drops(symbol: str, quote: dict, candles: list[dict], threshold_pct: float) -> list[dict]:
    alerts = []
    day_change = float(quote.get("day_change_pct") or 0.0)
    if day_change <= -threshold_pct:
        alerts.append({
            "symbol": symbol,
            "window": "day",
            "change_pct": _r(day_change),
            "threshold_pct": threshold_pct,
            "message": f"{symbol} fell {abs(day_change):.2f}% today (threshold {threshold_pct:g}%).",
        })

    if len(candles) >= 6:
        reference = float(candles[-6]["close"])
        week_change = (float(quote["last_price"]) / reference - 1) * 100
        if week_change <= -threshold_pct:
            alerts.append({
                "symbol": symbol,
                "window": "week",
                "change_pct": _r(week_change),
                "threshold_pct": threshold_pct,
                "message": f"{symbol} fell {abs(week_change):.2f}% over the last 5 sessions (threshold {threshold_pct:g}%).",
            })
    return alerts
