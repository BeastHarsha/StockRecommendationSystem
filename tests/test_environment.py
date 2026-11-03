import importlib

import pytest

pytestmark = pytest.mark.env

REQUIRED_MODULES = [
    "growwapi",
    "mcp",
    "mcp.server.fastmcp",
    "langgraph",
    "langchain_core",
    "langchain_openai",
    "langchain_mcp_adapters",
    "langchain_mcp_adapters.client",
    "chromadb",
    "sentence_transformers",
    "feedparser",
    "requests",
    "bs4",
    "pandas",
    "numpy",
    "pydantic",
    "dotenv",
    "apscheduler",
    "streamlit",
    "plotly",
    "zoneinfo",
]


@pytest.mark.parametrize("module", REQUIRED_MODULES)
def test_dependency_importable(module):
    importlib.import_module(module)


@pytest.mark.parametrize(
    "module",
    [
        "mcp_server.groww_server",
        "mcp_server.mock_data",
        "agent.analysis",
        "agent.assistant",
        "agent.graph",
        "agent.guardrails",
        "agent.mcp_client",
        "agent.schemas",
        "rag.ingest",
        "rag.retriever",
        "data.ipo_fetcher",
        "scheduler.monitor",
    ],
)
def test_project_modules_importable(module):
    importlib.import_module(module)
