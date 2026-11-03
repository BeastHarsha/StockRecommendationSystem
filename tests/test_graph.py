import asyncio
import dataclasses

import pytest
from langchain_core.messages import AIMessage

from agent.graph import run_with_toolbox
from agent.guardrails import GuardedToolbox
from config import load_settings
from data.ipo_fetcher import MockIpoSource
from rag.retriever import RetrievedDoc
from tests.conftest import make_tools

STEPS = ["fetch_portfolio", "fetch_history", "detect_price_drops", "retrieve_news", "summarize_performance",
         "self_check", "fetch_ipo_calendar", "estimate_listing_range", "compile_report"]


@pytest.fixture
def settings():
    return dataclasses.replace(load_settings(env_file=None), price_drop_threshold_pct=5.0, max_self_check_retries=2)


class PromptLLM:
    """Replies via a function of the prompt and records every prompt."""

    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    async def ainvoke(self, messages):
        prompt = messages[-1].content
        self.prompts.append(prompt)
        return AIMessage(content=self.reply(prompt))


class FailingLLM:
    async def ainvoke(self, messages):
        raise RuntimeError("API down")


def run(settings, **kwargs):
    return asyncio.run(run_with_toolbox(GuardedToolbox(make_tools(include_order_tool=True)), settings, **kwargs))


def test_report_without_llm_uses_grounded_templates(settings):
    report = run(settings, ipo_source=MockIpoSource())

    assert report.data_source == "mock" and report.llm_model is None
    assert len(report.holdings) == 5
    assert {(a.symbol, a.window) for a in report.alerts} == {("INFY", "day"), ("INFY", "week"), ("ITC", "week")}
    assert all(s.narrative_source == "template" and s.verified for s in report.summaries)
    assert [t.step for t in report.trace] == STEPS

    checks = [g.check for g in report.guardrails]
    assert checks.count("prediction_label") == 2 and "schema_validation" in checks
    allowlist = next(g for g in report.guardrails if g.check == "tool_allowlist")
    assert not allowlist.passed and "place_order" in allowlist.detail  # order tool dropped at load
    assert report.errors == []


def test_self_check_loop_regenerates_ungrounded_summaries(settings):
    llm = PromptLLM(lambda p: "The chart covers the period." if "previous draft" in p else "It returned 99.99%.")
    report = run(settings, llm=llm)

    assert all(s.verified and s.attempts == 2 and s.narrative_source == "llm" for s in report.summaries)
    assert sum("99.99%" in p for p in llm.prompts if "previous draft" in p) == 5
    assert [t.step for t in report.trace].count("summarize_performance") == 2
    grounded = [g.passed for g in report.guardrails if g.check == "groundedness"]
    assert grounded == [False] * 5 + [True] * 5


def test_self_check_loop_is_capped_and_flags_summaries(settings):
    report = run(settings, llm=PromptLLM(lambda p: "It returned 99.99%."))

    assert len(report.unverified_summaries) == 5
    assert all(s.attempts == 3 and s.ungrounded_numbers == ["99.99%"] for s in report.summaries)
    assert "partially unverified" in report.trace[-4].detail  # last self_check before the IPO steps
    assert [t.step for t in report.trace].count("summarize_performance") == 3


def test_llm_failure_falls_back_to_template(settings):
    report = run(settings, llm=FailingLLM())
    assert all(s.narrative_source == "template" and s.verified for s in report.summaries)
    assert len([e for e in report.errors if "used template instead" in e]) == 5


class FakeRetriever:
    def retrieve(self, symbol, query, k=5):
        if symbol != "INFY":
            return []
        return [RetrievedDoc(id="1", text="Infosys net profit rose 12% to Rs 7,364 crore.", title="Infosys Q2 profit rises 12%",
                             source="Mint", url="https://news/1", published_date="2026-10-31", distance=0.1)]


def test_news_grounding_and_citations(settings):
    ingested = []

    def ingest(symbol, name):
        ingested.append(symbol)
        return type("Result", (), {"removed_sentences": 2})()

    def reply(prompt):
        return "Infosys profit rose 12% [1], see also [9]." if "Infosys net profit" in prompt else "Prices moved."

    report = run(settings, llm=PromptLLM(reply), retriever=FakeRetriever(), ingest=ingest, refresh_news=True)

    infy = next(s for s in report.summaries if s.symbol == "INFY")
    assert infy.verified and [c.index for c in infy.citations] == [1]
    assert infy.citations[0].url == "https://news/1"
    assert [n.url for n in report.news["INFY"]] == ["https://news/1"] and report.news["TCS"] == []
    assert len(ingested) == 5
    sanitization = next(g for g in report.guardrails if g.check == "content_sanitization")
    assert "Removed 10" in sanitization.detail


def test_ipo_source_failure_is_reported(settings):
    class Broken:
        name = "broken"

        def fetch(self):
            raise ConnectionError("NSE unreachable")

    report = run(settings, ipo_source=Broken())
    assert report.ipos == []
    assert any("IPO calendar fetch failed" in e for e in report.errors)
