#!/usr/bin/env python3
"""
Test NeuroML tools

File: mcp_pkg/tests/test_neuroml_tools.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

import aiohttp
import pytest
import pytest_asyncio

from neuroml_mcp.tools.neuroml_tools import (
    get_models_from_neuromldb_tool,
    get_repositories_from_open_source_brain_tool,
)

logging.basicConfig(
    format="%(name)s (%(levelname)s) >>> %(message)s\n", level=logging.WARNING
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class MockContext(object):
    """Test stub replacing fastmcp.Context"""

    def __init__(self):
        self.lifespan_context = {}

    def set_state(self, key, val):
        self.lifespan_context[key] = val

    def get_state(self, arg):
        return self.lifespan_context.get(arg, None)


@pytest_asyncio.fixture
async def neuromldb_ctx():
    async with aiohttp.ClientSession() as ses:
        ctx = MockContext()
        ctx.set_state("neuromldb_session", ses)
        yield ctx


@pytest.mark.asyncio
async def test_get_models_from_neuromldb_download(neuromldb_ctx):
    model = "NMLCL000595"
    res = await get_models_from_neuromldb_tool(
        ctx=neuromldb_ctx, search_query=model, num=1, download=True
    )
    logger.debug(f"{res = }")
    assert len(res) == 1

    # Should download model
    assert model in res.keys()

    m = res[model]
    assert len(m["xml"]) != 0
    assert m["Type"] == "Cell"
    assert m["Publication_Year"] == 2015


@pytest.mark.asyncio
async def test_get_models_from_neuromldb_nodownload(neuromldb_ctx):
    model = "NMLCL000804"
    res = await get_models_from_neuromldb_tool(
        ctx=neuromldb_ctx, search_query=model, num=1, download=False
    )
    logger.debug(f"{res = }")
    assert len(res) == 1

    assert model in res.keys()

    m = res[model]
    assert len(m["xml"]) == 0
    assert m["Type"] == "Cell"
    assert m["Publication_Year"] == 2015


@pytest_asyncio.fixture
async def osbv2_ctx():
    async with aiohttp.ClientSession() as ses:
        ctx = MockContext()
        ctx.set_state("osbv2_session", ses)
        yield ctx


@pytest.mark.asyncio
async def test_get_repositories_from_open_source_brain(osbv2_ctx):
    # Test basic functionality with a simple search
    search_term = "cerebellum"
    res = await get_repositories_from_open_source_brain_tool(
        ctx=osbv2_ctx,
        search_query=search_term,
        search_data=True,
        search_models=True,
        num=2,
    )
    logger.debug(f"{res = }")

    # Should return a dictionary
    assert isinstance(res, dict)

    # Should have some results (may be empty depending on search)
    # Just checking it doesn't crash and returns proper structure
    assert "Error" not in res or isinstance(res["Error"], str)


@pytest.mark.asyncio
async def test_get_repositories_from_open_source_brain_no_results(osbv2_ctx):
    # Test with a search term that likely won't return results
    search_term = "nonexistent_search_term_12345"
    res = await get_repositories_from_open_source_brain_tool(
        ctx=osbv2_ctx,
        search_query=search_term,
        search_data=True,
        search_models=True,
        num=1,
    )
    logger.debug(f"{res = }")

    # Should return a dictionary
    assert isinstance(res, dict)

    # Should not crash, even if no results are found
    assert "Error" not in res or isinstance(res["Error"], str)
