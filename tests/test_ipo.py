import pytest
from pydantic import ValidationError

from agent.schemas import ListingEstimate
from data import ipo_fetcher
from data.ipo_fetcher import estimate_listing_range, parse_nse_date, parse_nse_issue, parse_price_band


@pytest.mark.parametrize("text, expected", [
    ("Rs.78 to Rs.82", (78.0, 82.0)),
    ("Rs.1,050 to Rs.1,105", (1050.0, 1105.0)),
    ("Rs.162", (162.0, 162.0)),
    ("-", (None, None)),
    (None, (None, None)),
])
def test_parse_price_band(text, expected):
    assert parse_price_band(text) == expected


def test_parse_nse_date():
    assert parse_nse_date("02-Nov-2026") == "2026-11-02"
    assert parse_nse_date("-") is None and parse_nse_date("bad") is None


def test_parse_nse_issue():
    issue = parse_nse_issue({
        "companyName": "Bombay Coated and Special Steels Limited", "symbol": "BCSSL", "series": "EQ", "status": "Closed",
        "issueStartDate": "29-Oct-2026", "issueEndDate": "02-Nov-2026", "issuePrice": "Rs.78 to Rs.82",
        "noOfTime": "32.733611395189214",
    })
    assert issue["price_high"] == 82.0 and issue["subscription_times"] == 32.73
    assert issue["issue_end"] == "2026-11-02"


def test_estimate_uses_subscription_band():
    estimate, note = estimate_listing_range({"price_high": 82.0, "subscription_times": 32.73, "status": "Closed", "series": "EQ"})
    assert note is None
    assert (estimate["low"], estimate["high"]) == (88.56, 106.6)
    assert estimate["gain_low_pct"] == 8.0 and estimate["gain_high_pct"] == 30.0
    ListingEstimate.model_validate(estimate)


@pytest.mark.parametrize("times, low_pct", [(0.5, -15.0), (1.0, -5.0), (5.0, 0.0), (20.0, 8.0), (50.0, 15.0), (250.0, 25.0)])
def test_estimate_band_boundaries(times, low_pct):
    estimate, _ = estimate_listing_range({"price_high": 100.0, "subscription_times": times})
    assert estimate["gain_low_pct"] == low_pct


def test_estimate_needs_price_and_subscription():
    assert estimate_listing_range({"price_high": None, "subscription_times": 10})[0] is None
    estimate, note = estimate_listing_range({"price_high": 100.0, "subscription_times": None})
    assert estimate is None and "subscription" in note


def test_estimate_flags_open_and_sme_issues():
    estimate, _ = estimate_listing_range({"price_high": 100.0, "subscription_times": 3, "status": "Active", "series": "SME"})
    joined = " ".join(estimate["assumptions"])
    assert "still open" in joined and "SME" in joined


def test_listing_estimate_requires_label_and_order():
    base = {"low": 90, "high": 110, "gain_low_pct": -10, "gain_high_pct": 10, "basis": "x", "assumptions": []}
    with pytest.raises(ValidationError):
        ListingEstimate.model_validate({**base, "label": "Guaranteed"})
    with pytest.raises(ValidationError):
        ListingEstimate.model_validate({**base, "low": 120})


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.headers = {}

    def get(self, url, timeout):
        if url == ipo_fetcher.NSE_UPCOMING:
            return FakeResponse([{"symbol": "ABC", "companyName": "ABC Ltd", "issuePrice": "Rs.10 to Rs.12", "status": "Forthcoming"}])
        if url == ipo_fetcher.NSE_CURRENT:
            return FakeResponse([
                {"symbol": "ABC", "companyName": "ABC Ltd", "issuePrice": "Rs.10 to Rs.12", "status": "Active", "category": "Total", "noOfTime": "4.5"},
                {"symbol": "ABC", "companyName": "ABC Ltd", "category": "Retail Individual Investors", "noOfTime": "9"},
            ])
        return FakeResponse({})


def test_nse_source_merges_and_prefers_current_totals(monkeypatch):
    monkeypatch.setattr(ipo_fetcher.requests, "Session", FakeSession)
    [issue] = ipo_fetcher.NseIpoSource().fetch()
    assert issue["status"] == "Active" and issue["subscription_times"] == 4.5
