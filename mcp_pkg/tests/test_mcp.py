#!/usr/bin/env python3
"""
Test MCP

File: test_mcp.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import pytest
from fastmcp.client import Client
from fastmcp.client.transports import FastMCPTransport

from neuroml_mcp.server.main import create_server


@pytest.fixture()
async def mcp_client():
    mcp = await create_server()
    async with Client(transport=mcp) as mcp_client:
        yield mcp_client


@pytest.mark.asyncio
async def test_list_tools(mcp_client: Client[FastMCPTransport]):
    all_tools = await mcp_client.list_tools()
    tool_names = [t.name for t in all_tools]
    assert "dummy_code_tool" in tool_names
