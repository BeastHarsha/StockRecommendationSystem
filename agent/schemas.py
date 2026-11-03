"""Pydantic schema for the report. compile_report validates against it before anything is shown."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from agent.guardrails import DISCLAIMER, PREDICTION_LABEL


class Holding(BaseModel):
    symbol: str
    company_name: str
    quantity: float
    average_price: float
    last_price: float | None
    day_change_pct: float | None
    invested: float
    current_value: float | None
    pnl: float | None
    pnl_pct: float | None


class Totals(BaseModel):
    invested: float
    current_value: float | None
    pnl: float | None
    pnl_pct: float | None


class PriceAlert(BaseModel):
    symbol: str
    window: Literal["day", "week"]
    change_pct: float
    threshold_pct: float
    message: str


class Metrics(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    start_close: float
    end_close: float
    period_high: float
    period_low: float
    return_pct: float
    max_drawdown_pct: float
    volatility_pct: float
    trading_days: int


class Citation(BaseModel):
    index: int
    title: str
    source: str
    url: str
    published_date: str


class Summary(BaseModel):
    symbol: str
    narrative: str
    narrative_source: Literal["llm", "template"]
    citations: list[Citation] = Field(default_factory=list)
    verified: bool
    ungrounded_numbers: list[str] = Field(default_factory=list)
    attempts: int = 1


class ListingEstimate(BaseModel):
    low: float
    high: float
    gain_low_pct: float
    gain_high_pct: float
    basis: str
    assumptions: list[str]
    label: str = PREDICTION_LABEL

    @model_validator(mode="after")
    def _ordered_and_labeled(self):
        if self.low > self.high:
            raise ValueError("Estimate low must not exceed high.")
        if self.label != PREDICTION_LABEL:
            raise ValueError("Every estimate must carry the prediction label.")
        return self


class Ipo(BaseModel):
    symbol: str
    company_name: str
    series: str
    status: str
    issue_start: str | None
    issue_end: str | None
    price_low: float | None
    price_high: float | None
    subscription_times: float | None
    estimate: ListingEstimate | None = None
    estimate_note: str | None = None


class GuardrailVerdict(BaseModel):
    check: str
    target: str
    passed: bool
    detail: str = ""


class TraceEvent(BaseModel):
    step: str
    detail: str
    data: dict = Field(default_factory=dict)


class Report(BaseModel):
    generated_at: datetime
    data_source: str
    llm_model: str | None
    holdings: list[Holding]
    totals: Totals
    alerts: list[PriceAlert]
    metrics: list[Metrics]
    summaries: list[Summary]
    news: dict[str, list[Citation]] = Field(default_factory=dict)
    history: dict[str, list[dict]]
    ipos: list[Ipo]
    guardrails: list[GuardrailVerdict]
    trace: list[TraceEvent]
    errors: list[str]
    disclaimer: str = DISCLAIMER

    @model_validator(mode="after")
    def _has_disclaimer(self):
        if self.disclaimer != DISCLAIMER:
            raise ValueError("Report must carry the standard disclaimer.")
        return self

    @property
    def unverified_summaries(self) -> list[Summary]:
        return [s for s in self.summaries if not s.verified]
