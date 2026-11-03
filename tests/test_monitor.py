import asyncio
import dataclasses
from datetime import datetime, timezone

import pytest

from agent.guardrails import GuardedToolbox
from config import load_settings
from scheduler import monitor
from scheduler.monitor import IST, AlertStore, is_market_open, poll_once, poll_with_toolbox
from tests.conftest import make_tools


@pytest.mark.parametrize("moment, expected", [
    (datetime(2026, 11, 3, 10, 0, tzinfo=IST), True),
    (datetime(2026, 11, 3, 9, 14, tzinfo=IST), False),
    (datetime(2026, 11, 3, 15, 30, tzinfo=IST), True),
    (datetime(2026, 11, 3, 15, 31, tzinfo=IST), False),
    (datetime(2026, 11, 7, 11, 0, tzinfo=IST), False),  # Saturday
    (datetime(2026, 11, 3, 5, 0, tzinfo=timezone.utc), True),  # 10:30 IST
])
def test_is_market_open(moment, expected):
    assert is_market_open(moment) is expected


def test_alert_store_dedupes_per_day(tmp_path):
    store = AlertStore(tmp_path / "alerts.json")
    alert = {"symbol": "INFY", "window": "day", "change_pct": -6.2, "threshold_pct": 5, "message": "m"}
    day1 = datetime(2026, 11, 3, 10, 0, tzinfo=IST)

    assert len(store.add([alert], day1)) == 1
    assert store.add([alert], day1) == []
    assert len(store.add([alert], datetime(2026, 11, 4, 10, 0, tzinfo=IST))) == 1
    assert [a["date"] for a in store.load()] == ["2026-11-03", "2026-11-04"]


def test_alert_store_recovers_from_corrupt_file(tmp_path):
    path = tmp_path / "alerts.json"
    path.write_text("{not json", encoding="utf-8")
    assert AlertStore(path).load() == []


def test_poll_finds_mock_drops():
    alerts = asyncio.run(poll_with_toolbox(GuardedToolbox(make_tools()), 5.0))
    assert {(a["symbol"], a["window"]) for a in alerts} == {("INFY", "day"), ("INFY", "week"), ("ITC", "week")}


def test_poll_skips_when_market_closed(tmp_path, monkeypatch):
    def fail():
        raise AssertionError("MCP server should not start outside market hours")

    monkeypatch.setattr(monitor, "groww_tools", fail)
    settings = dataclasses.replace(load_settings(env_file=None))
    saturday = datetime(2026, 11, 7, 11, 0, tzinfo=IST)
    assert asyncio.run(poll_once(settings, AlertStore(tmp_path / "a.json"), now=saturday)) == []
