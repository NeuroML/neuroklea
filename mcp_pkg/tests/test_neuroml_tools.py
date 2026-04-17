#!/usr/bin/env python3
"""
Test NeuroML tools

File: mcp_pkg/tests/test_neuroml_tools.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

import pytest

from neuroml_mcp.tools.neuroml_tools import get_models_from_neuromldb

logging.basicConfig(
    format="%(name)s (%(levelname)s) >>> %(message)s\n", level=logging.WARNING
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@pytest.mark.skip(reason="NeuroML-DB is currently down. Reported.")
@pytest.mark.asyncio
async def test_get_models_from_neuromldb():
    res = await get_models_from_neuromldb(
        search_query="granule cell", num=2, download=False
    )
    print(res)
