"""LangGraph state and nodes for the portfolio report.

fetch_portfolio -> fetch_history -> detect_price_drops -> retrieve_news -> summarize_performance
    -> self_check (loops back to summarize_performance for failing summaries, capped)
    -> fetch_ipo_calendar -> estimate_listing_range -> compile_report
"""

import asyncio
import json
import operator
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Annotated, Any, Callable, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from agent import analysis
from agent.guardrails import GuardedToolbox, ToolCallError, check_groundedness, numbers_in
from agent.mcp_client import groww_tools
from agent.schemas import Report
from data.ipo_fetcher import estimate_listing_range

HISTORY_DAYS = 180
NEWS_PER_HOLDING = 5
STRUCTURAL_NUMBERS = {5.0, 6.0, 52.0, 180.0}

SUMMARY_SYSTEM_PROMPT = """You summarise one stock's last six months for a portfolio report.
Rules:
- Use only numbers that appear in FACTS or SOURCES, written exactly as given. Never calculate, re-round or estimate a number.
- Cite news with [n] markers that match the SOURCES list. With no sources, describe only the price facts.
- No buy, sell or hold advice, no price targets and no predictions.
- Write 3 to 5 sentences of plain text."""


class ReportState(TypedDict, total=False):
    refresh_news: bool
    llm_unavailable: bool
    data_source: str
    holdings_raw: list[dict]
    quotes: dict[str, dict]
    holdings: list[dict]
    totals: dict
    history: dict[str, list[dict]]
    metrics: dict[str, dict]
    alerts: list[dict]
    news: dict[str, list[dict]]
    summaries: dict[str, dict]
    pending: list[str]
    retry_count: int
    ipos: list[dict]
    report: dict
    trace: Annotated[list[dict], operator.add]
    guardrails: Annotated[list[dict], operator.add]
    errors: Annotated[list[str], operator.add]


@dataclass
class ReportDeps:
    toolbox: GuardedToolbox
    threshold_pct: float
    max_retries: int
    llm: Any = None
    llm_model: str | None = None
    retriever: Any = None
    ingest: Callable[[str, str | None], Any] | None = None
    ipo_source: Any = None


def _event(step: str, detail: str, **data) -> dict:
    return {"step": step, "detail": detail, "data": data}


def _verdict(check: str, target: str, passed: bool, detail: str = "") -> dict:
    return {"check": check, "target": target, "passed": passed, "detail": detail}


def message_text(message) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    return "".join(
        b.get("text", "")
        for b in content
        if isinstance(b, dict) and b.get("type") in {"text", "output_text"}
    ).strip()


def billing_error_message(exc: Exception) -> str | None:
    """Return a short action for non-retryable OpenAI billing and quota errors."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        body = body.get("error", body)
    code = getattr(exc, "code", None) or (body.get("code") if isinstance(body, dict) else None)
    message = str(exc).lower()

    if code == "credit_balance_exhausted" or "no credits remaining" in message:
        return "OpenAI API credits are exhausted. Add credits in your OpenAI API billing settings."
    if code in {"organization_spend_limit_exceeded", "project_spend_limit_exceeded"}:
        return "OpenAI API spending limit reached. Check your organization or project spend limit."
    if code == "organization_usage_limit_exceeded":
        return "OpenAI API usage limit reached. Check your organization's usage limit."
    return None


def _facts(symbol: str, state: ReportState) -> dict:
    holding = next((h for h in state.get("holdings", []) if h["symbol"] == symbol), {})
    return {
        "company_name": holding.get("company_name", symbol),
        "last_price": holding.get("last_price"),
        "day_change_pct": holding.get("day_change_pct"),
        "average_buy_price": holding.get("average_price"),
        "position_pnl_pct": holding.get("pnl_pct"),
        **state.get("metrics", {}).get(symbol, {}),
    }


def template_narrative(facts: dict) -> str:
    position = "up" if (facts.get("position_pnl_pct") or 0) >= 0 else "down"
    return (
        f"Over the last six months ({facts['start_date']} to {facts['end_date']}), {facts['company_name']} moved from "
        f"₹{facts['start_close']:.2f} to ₹{facts['end_close']:.2f}, a return of {facts['return_pct']:.2f}%. "
        f"Its deepest fall from a peak was {facts['max_drawdown_pct']:.2f}% and annualised volatility was "
        f"{facts['volatility_pct']:.2f}%. It last traded at ₹{facts['last_price']:.2f} ({facts['day_change_pct']:.2f}% on the day), "
        f"and your position is {position} {abs(facts['position_pnl_pct']):.2f}% against your average buy price of "
        f"₹{facts['average_buy_price']:.2f}."
    )


def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return "(none)"
    return "\n".join(
        f"[{i}] {s['published_date']} | {s['source'] or 'unknown source'} | {s['text']}"
        for i, s in enumerate(sources, start=1)
    )


def _citations(narrative: str, sources: list[dict]) -> list[dict]:
    cited = sorted({int(n) for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", narrative) for n in re.split(r"\s*,\s*", group)})
    return [
        {"index": i, "title": sources[i - 1]["title"], "source": sources[i - 1]["source"],
         "url": sources[i - 1]["url"], "published_date": sources[i - 1]["published_date"]}
        for i in cited if 1 <= i <= len(sources)
    ]


def build_report_graph(deps: ReportDeps):
    toolbox = deps.toolbox

    async def fetch_portfolio(state: ReportState) -> dict:
        portfolio = await toolbox.call("get_portfolio")
        holdings = portfolio["holdings"]
        quotes, errors = {}, []
        for h in holdings:
            try:
                quotes[h["symbol"]] = await toolbox.call("get_quote", {"symbol": h["symbol"]})
            except ToolCallError as exc:
                errors.append(f"Quote for {h['symbol']} failed: {exc}")
        if holdings and len(errors) == len(holdings) and all("Access forbidden for this request" in error for error in errors):
            errors = [
                "Groww returned holdings but denied all live quote requests. Current value and P&L are unavailable. "
                "Check your Groww market-data access."
            ]
        rows = analysis.build_holdings(holdings, quotes)
        rejected = toolbox.rejected_at_load
        return {
            "data_source": portfolio.get("source", "unknown"),
            "holdings_raw": holdings,
            "quotes": quotes,
            "holdings": rows,
            "totals": analysis.portfolio_totals(rows),
            "errors": errors,
            "guardrails": [_verdict(
                "tool_allowlist", "MCP server tools", not rejected,
                f"Dropped non-allowlisted tools: {', '.join(rejected)}" if rejected
                else f"Only read-only tools loaded: {', '.join(t.name for t in toolbox.tools())}",
            )],
            "trace": [_event("fetch_portfolio", f"{len(holdings)} holdings, {len(quotes)} quotes", tool_calls=toolbox.drain())],
        }

    async def fetch_history(state: ReportState) -> dict:
        history, metrics, errors = {}, {}, []
        for symbol in state.get("quotes", {}):
            try:
                payload = await toolbox.call("get_historical_candles", {"symbol": symbol, "days": HISTORY_DAYS})
            except ToolCallError as exc:
                errors.append(f"History for {symbol} failed: {exc}")
                continue
            history[symbol] = payload["candles"]
            computed = analysis.compute_metrics(symbol, payload["candles"])
            if computed:
                metrics[symbol] = computed
        return {
            "history": history,
            "metrics": metrics,
            "errors": errors,
            "trace": [_event("fetch_history", f"6-month candles for {len(history)} holdings; metrics computed in Python",
                             tool_calls=toolbox.drain())],
        }

    async def detect_price_drops(state: ReportState) -> dict:
        alerts = []
        for symbol, quote in state.get("quotes", {}).items():
            alerts += analysis.detect_price_drops(symbol, quote, state.get("history", {}).get(symbol, []), deps.threshold_pct)
        return {
            "alerts": alerts,
            "trace": [_event("detect_price_drops", f"{len(alerts)} alerts at a {deps.threshold_pct:g}% threshold")],
        }

    async def retrieve_news(state: ReportState) -> dict:
        if deps.retriever is None:
            return {"news": {}, "trace": [_event("retrieve_news", "RAG disabled for this run")]}

        errors, verdicts, holdings = [], [], state.get("holdings", [])
        if state.get("refresh_news") and deps.ingest:
            removed = 0
            for h in holdings:
                try:
                    result = await asyncio.to_thread(deps.ingest, h["symbol"], h["company_name"])
                    removed += result.removed_sentences
                except Exception as exc:
                    errors.append(f"News ingest for {h['symbol']} failed: {exc}")
            verdicts.append(_verdict("content_sanitization", "ingested news", True,
                                     f"Removed {removed} instruction-like sentences before embedding"))

        news = {}
        for h in holdings:
            query = f"{h['company_name']} share price results performance news"
            try:
                docs = await asyncio.to_thread(deps.retriever.retrieve, h["symbol"], query, NEWS_PER_HOLDING)
                news[h["symbol"]] = [asdict(d) for d in docs]
            except Exception as exc:
                errors.append(f"News retrieval for {h['symbol']} failed: {exc}")
        found = sum(len(v) for v in news.values())
        return {
            "news": news,
            "errors": errors,
            "guardrails": verdicts,
            "trace": [_event("retrieve_news", f"{found} chunks retrieved across {len(news)} holdings",
                             refreshed=bool(state.get("refresh_news") and deps.ingest))],
        }

    async def summarize_performance(state: ReportState) -> dict:
        summaries = dict(state.get("summaries", {}))
        targets = state.get("pending") or list(state.get("metrics", {}))
        errors = []
        llm_unavailable = state.get("llm_unavailable", False)
        for symbol in targets:
            facts = _facts(symbol, state)
            sources = state.get("news", {}).get(symbol, [])
            previous = summaries.get(symbol, {})
            narrative, origin = None, "template"
            if deps.llm is not None and not llm_unavailable:
                prompt = f"FACTS:\n{json.dumps(facts, indent=2)}\n\nSOURCES:\n{_format_sources(sources)}"
                if previous.get("ungrounded_numbers"):
                    prompt += ("\n\nYour previous draft used numbers that are not in FACTS or SOURCES: "
                               f"{', '.join(previous['ungrounded_numbers'])}. Rewrite it without them.")
                try:
                    reply = await deps.llm.ainvoke([SystemMessage(SUMMARY_SYSTEM_PROMPT), HumanMessage(prompt)])
                    narrative, origin = message_text(reply), "llm"
                except Exception as exc:
                    if reason := billing_error_message(exc):
                        llm_unavailable = True
                        errors.append(f"{reason} Using template summaries for the rest of this report.")
                    else:
                        errors.append(f"LLM summary for {symbol} failed, used template instead: {exc}")
            if not narrative:
                narrative, origin = template_narrative(facts), "template"
            summaries[symbol] = {
                "symbol": symbol,
                "narrative": narrative,
                "narrative_source": origin,
                "citations": _citations(narrative, sources),
                "verified": False,
                "ungrounded_numbers": [],
                "attempts": previous.get("attempts", 0) + 1,
            }
        return {
            "summaries": summaries,
            "errors": errors,
            "llm_unavailable": llm_unavailable,
            "trace": [_event("summarize_performance", f"Drafted {len(targets)} summaries",
                             symbols=targets, attempt=state.get("retry_count", 0) + 1,
                             llm_unavailable=llm_unavailable)],
        }

    async def self_check(state: ReportState) -> dict:
        summaries = dict(state.get("summaries", {}))
        targets = state.get("pending") or list(summaries)
        verdicts, failing = [], []
        for symbol in targets:
            summary = dict(summaries[symbol])
            sources = state.get("news", {}).get(symbol, [])
            allowed = numbers_in([_facts(symbol, state)]) | numbers_in(
                [s["text"] for s in sources] + [s["title"] for s in sources]
            ) | STRUCTURAL_NUMBERS
            result = check_groundedness(summary["narrative"], allowed)
            summary["verified"] = result.passed
            summary["ungrounded_numbers"] = result.ungrounded
            summaries[symbol] = summary
            verdicts.append(_verdict(
                "groundedness", f"{symbol} summary (attempt {summary['attempts']})", result.passed,
                f"{result.checked} numbers checked" + (f"; ungrounded: {', '.join(result.ungrounded)}" if result.ungrounded else ""),
            ))
            if not result.passed:
                failing.append(symbol)

        retry_count = state.get("retry_count", 0)
        if failing and deps.llm is not None and retry_count < deps.max_retries:
            return {
                "summaries": summaries, "pending": failing, "retry_count": retry_count + 1, "guardrails": verdicts,
                "trace": [_event("self_check", f"{len(failing)} summaries failed grounding; retry {retry_count + 1} of {deps.max_retries}",
                                 failing=failing)],
            }
        detail = (f"Flagged as partially unverified after {retry_count} retries: {', '.join(failing)}"
                  if failing else "All summaries grounded")
        return {
            "summaries": summaries, "pending": [], "guardrails": verdicts,
            "trace": [_event("self_check", detail, failing=failing)],
        }

    def route_after_check(state: ReportState) -> str:
        return "summarize_performance" if state.get("pending") else "fetch_ipo_calendar"

    async def fetch_ipo_calendar(state: ReportState) -> dict:
        if deps.ipo_source is None:
            return {"ipos": [], "trace": [_event("fetch_ipo_calendar", "IPO tracking disabled for this run")]}
        try:
            ipos = await asyncio.to_thread(deps.ipo_source.fetch)
        except Exception as exc:
            return {"ipos": [], "errors": [f"IPO calendar fetch failed: {exc}"],
                    "trace": [_event("fetch_ipo_calendar", "IPO source unavailable")]}
        return {"ipos": ipos, "trace": [_event("fetch_ipo_calendar", f"{len(ipos)} IPOs from {deps.ipo_source.name}")]}

    async def estimate_listing(state: ReportState) -> dict:
        ipos, verdicts = [], []
        for ipo in state.get("ipos", []):
            estimate, note = estimate_listing_range(ipo)
            ipos.append({**ipo, "estimate": estimate, "estimate_note": note})
            if estimate:
                verdicts.append(_verdict("prediction_label", f"{ipo['symbol']} listing estimate", True,
                                         "Range, assumptions and 'not financial advice' label attached"))
        estimated = sum(1 for i in ipos if i["estimate"])
        return {"ipos": ipos, "guardrails": verdicts,
                "trace": [_event("estimate_listing_range", f"Estimated {estimated} of {len(ipos)} IPOs with the subscription heuristic")]}

    async def compile_report(state: ReportState) -> dict:
        summaries = list(state.get("summaries", {}).values())
        report = Report.model_validate({
            "generated_at": datetime.now(timezone.utc),
            "data_source": state.get("data_source", "unknown"),
            "llm_model": deps.llm_model if deps.llm is not None else None,
            "holdings": state.get("holdings", []),
            "totals": state.get("totals", analysis.portfolio_totals([])),
            "alerts": state.get("alerts", []),
            "metrics": list(state.get("metrics", {}).values()),
            "summaries": summaries,
            "news": {
                symbol: [{"index": i, "title": d["title"], "source": d["source"], "url": d["url"],
                          "published_date": d["published_date"]} for i, d in enumerate(docs, start=1)]
                for symbol, docs in state.get("news", {}).items()
            },
            "history": state.get("history", {}),
            "ipos": state.get("ipos", []),
            "guardrails": state.get("guardrails", []) + [
                _verdict("schema_validation", "report", True, "Report matches the Pydantic schema and carries the disclaimer")
            ],
            "trace": state.get("trace", []) + [_event("compile_report", "Report validated")],
            "errors": state.get("errors", []),
        })
        return {"report": report.model_dump(mode="json")}

    graph = StateGraph(ReportState)
    graph.add_node("fetch_portfolio", fetch_portfolio)
    graph.add_node("fetch_history", fetch_history)
    graph.add_node("detect_price_drops", detect_price_drops)
    graph.add_node("retrieve_news", retrieve_news)
    graph.add_node("summarize_performance", summarize_performance)
    graph.add_node("self_check", self_check)
    graph.add_node("fetch_ipo_calendar", fetch_ipo_calendar)
    graph.add_node("estimate_listing_range", estimate_listing)
    graph.add_node("compile_report", compile_report)

    graph.add_edge(START, "fetch_portfolio")
    graph.add_edge("fetch_portfolio", "fetch_history")
    graph.add_edge("fetch_history", "detect_price_drops")
    graph.add_edge("detect_price_drops", "retrieve_news")
    graph.add_edge("retrieve_news", "summarize_performance")
    graph.add_edge("summarize_performance", "self_check")
    graph.add_conditional_edges("self_check", route_after_check, ["summarize_performance", "fetch_ipo_calendar"])
    graph.add_edge("fetch_ipo_calendar", "estimate_listing_range")
    graph.add_edge("estimate_listing_range", "compile_report")
    graph.add_edge("compile_report", END)
    return graph.compile()


def make_llm(settings):
    if not settings.openai_api_key:
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        use_responses_api=True,
        reasoning_effort="none",
        max_tokens=600,
        timeout=60,
        max_retries=0,
    )


async def run_with_toolbox(toolbox: GuardedToolbox, settings, *, llm=None, retriever=None, ingest=None,
                           ipo_source=None, refresh_news: bool = False) -> Report:
    deps = ReportDeps(
        toolbox=toolbox,
        threshold_pct=settings.price_drop_threshold_pct,
        max_retries=settings.max_self_check_retries,
        llm=llm,
        llm_model=settings.llm_model,
        retriever=retriever,
        ingest=ingest,
        ipo_source=ipo_source,
    )
    state = await build_report_graph(deps).ainvoke(
        {"refresh_news": refresh_news, "retry_count": 0, "trace": [], "guardrails": [], "errors": []}
    )
    return Report.model_validate(state["report"])


async def run_report(settings, **kwargs) -> Report:
    """Run the full report against the read-only Groww MCP server."""
    async with groww_tools() as tools:
        return await run_with_toolbox(GuardedToolbox(tools), settings, **kwargs)
