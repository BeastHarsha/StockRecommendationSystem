import pytest

from agent.analysis import build_holdings, compute_metrics, detect_price_drops, portfolio_totals


def candles(closes):
    return [{"date": f"2026-01-{i + 1:02d}", "close": c} for i, c in enumerate(closes)]


def test_build_holdings_and_totals():
    rows = build_holdings(
        [{"symbol": "A", "company_name": "Alpha", "quantity": 10, "average_price": 100}],
        {"A": {"last_price": 110, "day_change_pct": -1.5}},
    )
    assert rows[0]["pnl"] == 100.0 and rows[0]["pnl_pct"] == 10.0
    assert portfolio_totals(rows) == {"invested": 1000.0, "current_value": 1100.0, "pnl": 100.0, "pnl_pct": 10.0}


def test_build_holdings_skips_missing_quotes():
    assert build_holdings([{"symbol": "A", "quantity": 1, "average_price": 1}], {}) == []


def test_compute_metrics():
    m = compute_metrics("A", candles([100, 120, 90, 110]))
    assert m["return_pct"] == 10.0
    assert m["max_drawdown_pct"] == -25.0
    assert m["period_high"] == 120 and m["period_low"] == 90
    assert m["volatility_pct"] > 0


def test_compute_metrics_needs_two_candles():
    assert compute_metrics("A", candles([100])) is None


@pytest.mark.parametrize("day_change, expected", [(-5.0, True), (-4.99, False), (3.0, False)])
def test_day_drop_threshold(day_change, expected):
    alerts = detect_price_drops("A", {"day_change_pct": day_change, "last_price": 100}, [], 5)
    assert any(a["window"] == "day" for a in alerts) is expected


def test_week_drop_uses_close_five_sessions_back():
    history = candles([100, 100, 100, 100, 100, 100])
    alerts = detect_price_drops("A", {"day_change_pct": -1, "last_price": 94}, history, 5)
    assert [a["window"] for a in alerts] == ["week"]
    assert alerts[0]["change_pct"] == -6.0
