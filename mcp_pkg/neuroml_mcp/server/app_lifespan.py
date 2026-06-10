#!/usr/bin/env python3
"""
Lifespan for MCP server

File: mcp_pkg/neuroml_mcp/server/lifespan.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

import aiohttp
from fastmcp.server.lifespan import lifespan

from ..utils import cleanup_cache_dir, init_cache_dir

logging.basicConfig(
    format="%(name)s (%(levelname)s) >>> %(message)s\n", level=logging.WARNING
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@lifespan
async def app_lifespan(server):
    """Life span for server"""
    logger.info("MCP Server starting up")

    # add more sessions here as required
    neuromldb_session = aiohttp.ClientSession()
    init_cache_dir()

    try:
        yield {"neuromldb_session": neuromldb_session}
    finally:
        logger.info("MCP Server shutting down")

        await neuromldb_session.close()
        cleanup_cache_dir()

        logger.info("MCP Server shut down")
