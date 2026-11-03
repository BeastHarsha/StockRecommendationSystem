"""APScheduler job that polls quotes for price drops during market hours.

Run with:
    python -m scheduler.monitor            poll every POLL_INTERVAL_MINUTES during market hours
    python -m scheduler.monitor --once     poll once now (still respects market hours)
    python -m scheduler.monitor --once --force   poll once now, even outside market hours
"""

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from agent.analysis import detect_price_drops
from agent.guardrails import GuardedToolbox, ToolCallError
from agent.mcp_client import groww_tools
from config import ROOT_DIR, load_settings

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
ALERTS_PATH = ROOT_DIR / "storage" / "alerts.json"
ALERT_HISTORY_LIMIT = 500

log = logging.getLogger("monitor")


def is_market_open(now: datetime) -> bool:
    """NSE regular session, Monday to Friday. Exchange holidays are not modelled."""
    local = now.astimezone(IST)
    return local.weekday() < 5 and MARKET_OPEN <= local.time() <= MARKET_CLOSE


class AlertStore:
    """Alerts persisted as JSON so the dashboard can read them. One alert per symbol, window and day."""

    def __init__(self, path: Path = ALERTS_PATH):
        self.path = path

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("Alert file %s is corrupt; starting fresh", self.path)
            return []

    def add(self, alerts: list[dict], now: datetime) -> list[dict]:
        existing = self.load()
        seen = {(a["symbol"], a["window"], a["date"]) for a in existing}
        today = now.astimezone(IST).date().isoformat()
        new = []
        for alert in alerts:
            key = (alert["symbol"], alert["window"], today)
            if key in seen:
                continue
            seen.add(key)
            new.append({**alert, "date": today, "detected_at": now.astimezone(IST).isoformat(timespec="seconds")})
        if new:
            self._save((existing + new)[-ALERT_HISTORY_LIMIT:])
        return new

    def _save(self, alerts: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(alerts, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)


async def poll_with_toolbox(toolbox: GuardedToolbox, threshold_pct: float) -> list[dict]:
    portfolio = await toolbox.call("get_portfolio")
    found = []
    for holding in portfolio["holdings"]:
        symbol = holding["symbol"]
        try:
            quote = await toolbox.call("get_quote", {"symbol": symbol})
            candles = (await toolbox.call("get_historical_candles", {"symbol": symbol, "days": 14}))["candles"]
        except ToolCallError as exc:
            log.warning("Skipping %s: %s", symbol, exc)
            continue
        found += detect_price_drops(symbol, quote, candles, threshold_pct)
    return found


async def poll_once(settings, store: AlertStore, now: datetime | None = None, force: bool = False) -> list[dict]:
    now = now or datetime.now(IST)
    if not force and not is_market_open(now):
        log.info("Market closed at %s; skipping poll", now.astimezone(IST).strftime("%a %H:%M"))
        return []
    async with groww_tools() as tools:
        found = await poll_with_toolbox(GuardedToolbox(tools), settings.price_drop_threshold_pct)
    new = store.add(found, now)
    for alert in new:
        log.warning("ALERT: %s", alert["message"])
    log.info("Poll done: %d drops found, %d new alerts", len(found), len(new))
    return new


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Poll the portfolio for price drops.")
    parser.add_argument("--once", action="store_true", help="poll once and exit")
    parser.add_argument("--force", action="store_true", help="poll even outside market hours")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = load_settings()
    store = AlertStore()

    if args.once:
        asyncio.run(poll_once(settings, store, force=args.force))
        return

    from apscheduler.schedulers.blocking import BlockingScheduler

    def job() -> None:
        try:
            asyncio.run(poll_once(settings, store, force=args.force))
        except Exception:
            log.exception("Poll failed; will retry on the next interval")

    scheduler = BlockingScheduler(timezone=IST)
    scheduler.add_job(job, "interval", minutes=settings.poll_interval_minutes,
                      next_run_time=datetime.now(IST), max_instances=1, coalesce=True)
    log.info("Monitoring every %d minutes (threshold %g%%). Ctrl+C to stop.",
             settings.poll_interval_minutes, settings.price_drop_threshold_pct)
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("Monitor stopped")


if __name__ == "__main__":
    main()
