import importlib

import pytest

REQUIRED_MODULES = [
    "growwapi",
    "mcp",
    "langgraph",
    "langchain_core",
    "langchain_anthropic",
    "langchain_mcp_adapters",
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
]


@pytest.mark.parametrize("module", REQUIRED_MODULES)
def test_dependency_importable(module):
    importlib.import_module(module)


@pytest.mark.parametrize(
    "module",
    [
        "mcp_server.groww_server",
        "agent.graph",
        "agent.guardrails",
        "rag.ingest",
        "rag.retriever",
        "data.ipo_fetcher",
        "scheduler.monitor",
    ],
)
def test_project_modules_importable(module):
    importlib.import_module(module)
