"""Fetch the current and upcoming IPO calendar from NSE, and estimate listing ranges with a transparent heuristic."""

import re
from datetime import date, datetime, timedelta

import requests

from agent.guardrails import PREDICTION_LABEL

NSE_HOME = "https://www.nseindia.com/"
NSE_CURRENT = "https://www.nseindia.com/api/ipo-current-issue"
NSE_UPCOMING = "https://www.nseindia.com/api/all-upcoming-issues?category=ipo"
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/market-data/all-upcoming-issues-ipo",
}

# Expected listing gain bands by subscription multiple: (minimum times subscribed, low %, high %).
# A simple, explainable heuristic, not a trained model.
SUBSCRIPTION_BANDS = [
    (100.0, 25.0, 70.0),
    (50.0, 15.0, 45.0),
    (20.0, 8.0, 30.0),
    (5.0, 0.0, 15.0),
    (1.0, -5.0, 8.0),
    (0.0, -15.0, 0.0),
]


def parse_price_band(text: str | None) -> tuple[float | None, float | None]:
    prices = [float(p.replace(",", "")) for p in re.findall(r"\d[\d,]*(?:\.\d+)?", text or "")]
    if not prices:
        return None, None
    return min(prices), max(prices)


def parse_nse_date(text: str | None) -> str | None:
    if not text or text.strip() in {"-", ""}:
        return None
    try:
        return datetime.strptime(text.strip(), "%d-%b-%Y").date().isoformat()
    except ValueError:
        return None


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_nse_issue(raw: dict) -> dict:
    low, high = parse_price_band(raw.get("issuePrice"))
    times = _to_float(raw.get("noOfTime"))
    return {
        "symbol": raw.get("symbol") or "",
        "company_name": raw.get("companyName") or raw.get("symbol") or "",
        "series": raw.get("series") or "",
        "status": raw.get("status") or "",
        "issue_start": parse_nse_date(raw.get("issueStartDate")),
        "issue_end": parse_nse_date(raw.get("issueEndDate")),
        "price_low": low,
        "price_high": high,
        "subscription_times": round(times, 2) if times is not None else None,
    }


class NseIpoSource:
    name = "NSE"

    def __init__(self, timeout: int = 15):
        self.timeout = timeout

    def fetch(self) -> list[dict]:
        session = requests.Session()
        session.headers.update(BROWSER_HEADERS)
        session.get(NSE_HOME, timeout=self.timeout)  # NSE sets the cookies its API needs on the home page.

        issues: dict[str, dict] = {}
        for url in (NSE_UPCOMING, NSE_CURRENT):  # current issues carry subscription data, so they win
            response = session.get(url, timeout=self.timeout)
            response.raise_for_status()
            for raw in response.json():
                if raw.get("category") not in (None, "Total"):
                    continue
                issue = parse_nse_issue(raw)
                if issue["symbol"]:
                    issues[issue["symbol"]] = issue
        return list(issues.values())


class MockIpoSource:
    name = "mock"

    def fetch(self) -> list[dict]:
        today = date.today()
        return [
            {"symbol": "SAMPLEFIN", "company_name": "Sample Finance Limited", "series": "EQ", "status": "Closed",
             "issue_start": (today - timedelta(days=5)).isoformat(), "issue_end": (today - timedelta(days=2)).isoformat(),
             "price_low": 78.0, "price_high": 82.0, "subscription_times": 32.73},
            {"symbol": "DEMOTECH", "company_name": "Demo Technologies Limited", "series": "EQ", "status": "Active",
             "issue_start": (today - timedelta(days=1)).isoformat(), "issue_end": (today + timedelta(days=2)).isoformat(),
             "price_low": 340.0, "price_high": 357.0, "subscription_times": 1.8},
            {"symbol": "EXAMPLESME", "company_name": "Example Agro SME Limited", "series": "SME", "status": "Forthcoming",
             "issue_start": (today + timedelta(days=4)).isoformat(), "issue_end": (today + timedelta(days=7)).isoformat(),
             "price_low": 120.0, "price_high": 126.0, "subscription_times": None},
        ]


def estimate_listing_range(ipo: dict) -> tuple[dict | None, str | None]:
    """Return (estimate, note). The estimate is None when there isn't enough data."""
    price = ipo.get("price_high")
    times = ipo.get("subscription_times")
    if price is None:
        return None, "The source didn't provide a price band for this issue."
    if times is None:
        return None, "No subscription data yet; an estimate needs at least one day of bidding."

    min_times, gain_low, gain_high = next(band for band in SUBSCRIPTION_BANDS if times >= band[0])
    assumptions = [
        f"Based only on the overall subscription of {times:g}x at the upper price band of ₹{price:g}.",
        f"Issues subscribed at least {min_times:g}x have been treated as listing between {gain_low:+g}% and {gain_high:+g}%.",
        "Ignores grey-market premium, sector sentiment, market conditions on listing day and allotment.",
    ]
    if ipo.get("status", "").lower() == "active":
        assumptions.append("The issue is still open, so the subscription figure can change.")
    if ipo.get("series", "").upper() == "SME":
        assumptions.append("SME listings are thinly traded and can move far outside this range.")

    estimate = {
        "low": round(price * (1 + gain_low / 100), 2),
        "high": round(price * (1 + gain_high / 100), 2),
        "gain_low_pct": gain_low,
        "gain_high_pct": gain_high,
        "basis": "Subscription-multiple heuristic",
        "assumptions": assumptions,
        "label": PREDICTION_LABEL,
    }
    return estimate, None
