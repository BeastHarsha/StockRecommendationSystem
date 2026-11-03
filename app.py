import asyncio
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agent.guardrails import DISCLAIMER
from config import Settings, load_settings
from scheduler.monitor import AlertStore

st.set_page_config(page_title="Portfolio Pulse", page_icon="📈", layout="wide")

RUPEE = "₹%.2f"
PERCENT = "%.2f%%"


@st.cache_resource(show_spinner=False)
def get_rag(persist_dir: str, model_name: str):
    from rag.ingest import SentenceTransformerEmbedder, ingest_news, open_collection
    from rag.retriever import NewsRetriever

    collection = open_collection(Path(persist_dir))
    embedder = SentenceTransformerEmbedder(model_name)
    return NewsRetriever(collection, embedder), lambda symbol, name: ingest_news(collection, embedder, symbol, name)


@st.cache_resource(show_spinner=False)
def get_llm(settings: Settings):
    from agent.graph import make_llm

    return make_llm(settings)


def generate_report(settings: Settings, use_rag: bool, refresh_news: bool, ipo_choice: str):
    from agent.graph import run_report
    from data.ipo_fetcher import MockIpoSource, NseIpoSource

    retriever, ingest = get_rag(str(settings.chroma_persist_dir), settings.embedding_model) if use_rag else (None, None)
    ipo_source = {"NSE (live)": NseIpoSource(), "Sample data": MockIpoSource()}.get(ipo_choice)
    return asyncio.run(run_report(
        settings,
        llm=get_llm(settings),
        retriever=retriever,
        ingest=ingest,
        ipo_source=ipo_source,
        refresh_news=refresh_news,
    ))


def sidebar(settings: Settings) -> None:
    with st.sidebar:
        st.header("Setup")
        st.markdown(f"**Data source:** {'Mock data' if settings.use_mock_groww else 'Groww API (read-only)'}")
        st.markdown(f"**LLM:** `{settings.llm_model}`" if settings.anthropic_api_key else "**LLM:** not configured, using templates")
        st.markdown(f"**Price-drop threshold:** {settings.price_drop_threshold_pct:g}%")
        missing = settings.missing_credentials()
        if missing:
            st.warning("Missing in .env: " + ", ".join(missing))

        st.header("Report options")
        use_rag = st.checkbox("Ground summaries in news (RAG)", value=True,
                              help="The first run downloads the embedding model and can take a minute.")
        refresh_news = st.checkbox("Fetch fresh news first", value=False, disabled=not use_rag,
                                   help="Pulls recent headlines from Google News into the vector store.")
        ipo_choice = st.radio("IPO data", ["NSE (live)", "Sample data", "Off"], horizontal=True)

        if st.button("Generate report", type="primary", width="stretch"):
            with st.spinner("Running the agent..."):
                try:
                    st.session_state["report"] = generate_report(settings, use_rag, refresh_news and use_rag, ipo_choice)
                    st.session_state.pop("report_error", None)
                except Exception as exc:
                    st.session_state["report_error"] = str(exc)


def portfolio_tab(report) -> None:
    totals = report.totals
    c1, c2, c3 = st.columns(3)
    c1.metric("Invested", f"₹{totals.invested:,.2f}")
    c2.metric("Current value", f"₹{totals.current_value:,.2f}")
    c3.metric("Unrealised P&L", f"₹{totals.pnl:,.2f}", f"{totals.pnl_pct:.2f}%")

    df = pd.DataFrame([h.model_dump() for h in report.holdings])
    st.dataframe(
        df[["symbol", "company_name", "quantity", "average_price", "last_price", "day_change_pct", "invested", "current_value", "pnl", "pnl_pct"]],
        hide_index=True,
        width="stretch",
        column_config={
            "symbol": "Symbol",
            "company_name": "Company",
            "quantity": st.column_config.NumberColumn("Qty", format="%g"),
            "average_price": st.column_config.NumberColumn("Avg price", format=RUPEE),
            "last_price": st.column_config.NumberColumn("Last price", format=RUPEE),
            "day_change_pct": st.column_config.NumberColumn("Day %", format=PERCENT),
            "invested": st.column_config.NumberColumn("Invested", format=RUPEE),
            "current_value": st.column_config.NumberColumn("Value", format=RUPEE),
            "pnl": st.column_config.NumberColumn("P&L", format=RUPEE),
            "pnl_pct": st.column_config.NumberColumn("P&L %", format=PERCENT),
        },
    )
    if report.errors:
        with st.expander(f"{len(report.errors)} issues during this run"):
            for error in report.errors:
                st.write(f"- {error}")


def alerts_tab(report) -> None:
    st.subheader("From this report")
    if report is None:
        st.caption("Generate a report to check for price drops.")
    elif not report.alerts:
        st.success("No holding has dropped past the threshold.")
    for alert in report.alerts if report else []:
        st.error(alert.message, icon="📉")

    st.subheader("From the background monitor")
    stored = AlertStore().load()
    if stored:
        st.dataframe(pd.DataFrame(reversed(stored))[["detected_at", "symbol", "window", "change_pct", "message"]],
                     hide_index=True, width="stretch")
    else:
        st.caption("No monitor alerts yet. Start it with `python -m scheduler.monitor`.")


def performance_tab(report) -> None:
    if not report.metrics:
        st.info("No price history was available.")
        return
    symbol = st.selectbox("Holding", [m.symbol for m in report.metrics])
    metrics = next(m for m in report.metrics if m.symbol == symbol)
    summary = next((s for s in report.summaries if s.symbol == symbol), None)

    c1, c2, c3 = st.columns(3)
    c1.metric("6-month return", f"{metrics.return_pct:.2f}%")
    c2.metric("Max drawdown", f"{metrics.max_drawdown_pct:.2f}%")
    c3.metric("Annualised volatility", f"{metrics.volatility_pct:.2f}%")

    candles = pd.DataFrame(report.history.get(symbol, []))
    if not candles.empty:
        fig = go.Figure(go.Candlestick(x=candles["date"], open=candles["open"], high=candles["high"],
                                       low=candles["low"], close=candles["close"], name=symbol))
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False,
                          yaxis_title="Price (₹)")
        st.plotly_chart(fig, width="stretch")

    if summary:
        if summary.verified:
            st.success("Every number in this summary matches the data or a cited source.", icon="✅")
        else:
            st.warning("Partially unverified: these numbers weren't found in the data or sources: "
                       + ", ".join(summary.ungrounded_numbers), icon="⚠️")
        st.write(summary.narrative)
        origin = f"Written by {report.llm_model}" if summary.narrative_source == "llm" else "Written from a template (no LLM)"
        st.caption(f"{origin} · attempt {summary.attempts}")
        if summary.citations:
            st.markdown("**Sources**")
            for c in summary.citations:
                st.markdown(f"[{c.index}] [{c.title}]({c.url}) · {c.source} · {c.published_date}")

    related = report.news.get(symbol, [])
    if related:
        with st.expander(f"Retrieved news ({len(related)})"):
            for item in related:
                st.markdown(f"- [{item.title}]({item.url}) · {item.source} · {item.published_date}")


def ipo_tab(report) -> None:
    if not report.ipos:
        st.info("No IPO data in this report.")
        return
    st.caption("Listing ranges are illustrative estimates from a simple subscription heuristic, not financial advice.")
    for ipo in report.ipos:
        with st.container(border=True):
            st.markdown(f"**{ipo.company_name}** · `{ipo.symbol}` · {ipo.series} · {ipo.status}")
            c1, c2, c3 = st.columns(3)
            band = f"₹{ipo.price_low:g} – ₹{ipo.price_high:g}" if ipo.price_high else "Not available"
            c1.metric("Price band", band)
            c2.metric("Subscribed", f"{ipo.subscription_times:g}x" if ipo.subscription_times is not None else "Not yet")
            c3.metric("Issue dates", f"{ipo.issue_start or '?'} → {ipo.issue_end or '?'}")
            if ipo.estimate:
                e = ipo.estimate
                st.info(f"**Estimated listing range: ₹{e.low:g} – ₹{e.high:g}** "
                        f"({e.gain_low_pct:+g}% to {e.gain_high_pct:+g}%). {e.label}")
                with st.expander("Assumptions"):
                    for assumption in e.assumptions:
                        st.write(f"- {assumption}")
            else:
                st.caption(f"No estimate: {ipo.estimate_note}")


def ask_tab(settings: Settings) -> None:
    llm = get_llm(settings)
    if llm is None:
        st.info("Add ANTHROPIC_API_KEY to .env to ask questions. The assistant uses the same read-only tools.")
        return
    st.caption("The assistant can only call get_portfolio, get_quote and get_historical_candles.")
    question = st.text_input("Ask about your portfolio", placeholder="Which holding fell the most today?")
    if st.button("Ask", disabled=not question):
        from agent.assistant import ask

        with st.spinner("Calling tools..."):
            try:
                st.session_state["answer"] = asyncio.run(ask(question, llm))
            except Exception as exc:
                st.error(f"The assistant failed: {exc}")

    result = st.session_state.get("answer")
    if result:
        if not result.verified:
            st.warning("Unverified numbers in this answer: " + ", ".join(result.ungrounded), icon="⚠️")
        st.markdown(result.answer)
        with st.expander(f"{len(result.tool_calls)} tool calls · {result.steps} steps · {result.stopped}"):
            st.dataframe(pd.DataFrame(result.tool_calls), hide_index=True, width="stretch")


def trace_tab(report) -> None:
    st.subheader("Guardrail checks")
    verdicts = pd.DataFrame([g.model_dump() for g in report.guardrails])
    verdicts["passed"] = verdicts["passed"].map({True: "✅ passed", False: "⚠️ failed"})
    st.dataframe(verdicts[["check", "target", "passed", "detail"]], hide_index=True, width="stretch")

    st.subheader("Agent steps")
    for i, event in enumerate(report.trace, start=1):
        with st.expander(f"{i}. {event.step} — {event.detail}"):
            tool_calls = event.data.get("tool_calls")
            if tool_calls:
                st.dataframe(pd.DataFrame(tool_calls), hide_index=True, width="stretch")
            other = {k: v for k, v in event.data.items() if k != "tool_calls"}
            if other:
                st.json(other)
            if not event.data:
                st.caption("No extra data.")


def main() -> None:
    st.title("Portfolio Pulse")
    st.caption(DISCLAIMER)

    try:
        settings = load_settings()
    except ValueError as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()

    sidebar(settings)
    if error := st.session_state.get("report_error"):
        st.error(f"Report failed: {error}")

    report = st.session_state.get("report")
    if report is None:
        st.info("Choose your options in the sidebar and click **Generate report**.")
    else:
        st.caption(f"Generated {report.generated_at.astimezone():%d %b %Y, %H:%M} · data: {report.data_source}")

    tabs = st.tabs(["Portfolio", "Alerts", "Performance", "IPOs", "Ask", "Trace"])
    with tabs[1]:
        alerts_tab(report)
    with tabs[4]:
        ask_tab(settings)
    if report is None:
        return
    with tabs[0]:
        portfolio_tab(report)
    with tabs[2]:
        performance_tab(report)
    with tabs[3]:
        ipo_tab(report)
    with tabs[5]:
        trace_tab(report)


main()
