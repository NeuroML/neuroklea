#!/usr/bin/env python3
"""
Test code related tools

File: mcp_pkg/tests/test_code_tools.py

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
async def test_dummy_code_tool(mcp_client: Client[FastMCPTransport]):
    ret = await mcp_client.call_tool_mcp(
        "dummy_code_tool", arguments={"astring": "Hello world"}
    )
    assert ret.structuredContent["result"] == "I got Hello world"
