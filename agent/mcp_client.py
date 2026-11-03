"""Connects to the read-only Groww MCP server over stdio and loads its tools for LangChain."""

import os
import sys
from contextlib import asynccontextmanager

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from config import ROOT_DIR

SERVER_NAME = "groww"


def server_connection() -> dict:
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", "mcp_server.groww_server"],
        "cwd": str(ROOT_DIR),
        "env": dict(os.environ),
    }


@asynccontextmanager
async def groww_tools():
    """Yield the server's tools over one session, so the server process starts once per run."""
    client = MultiServerMCPClient({SERVER_NAME: server_connection()})
    async with client.session(SERVER_NAME) as session:
        yield await load_mcp_tools(session)
