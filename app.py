import streamlit as st

from config import load_settings

st.set_page_config(page_title="Portfolio Pulse", page_icon="📈", layout="wide")

st.title("Portfolio Pulse")
st.caption(
    "Read-only portfolio monitoring. Nothing shown here is financial advice; "
    "all predictions are illustrative estimates."
)

try:
    settings = load_settings()
except ValueError as exc:
    st.error(f"Configuration error: {exc}")
    st.stop()

st.subheader("Setup status")

col1, col2, col3 = st.columns(3)
col1.metric("Data source", "Mock data" if settings.use_mock_groww else "Groww API")
col2.metric("LLM model", settings.llm_model)
col3.metric("Price-drop threshold", f"{settings.price_drop_threshold_pct:g}%")

missing = settings.missing_credentials()
if missing:
    st.warning(
        "Missing credentials in .env: "
        + ", ".join(missing)
        + ". Copy .env.example to .env and fill them in."
    )
else:
    st.success("All required credentials are set.")

st.info("Portfolio, alerts, IPO and trace views will appear here as each phase lands.")
