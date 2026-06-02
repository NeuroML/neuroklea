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

logging.basicConfig(
    format="%(name)s (%(levelname)s) >>> %(message)s\n", level=logging.WARNING
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@lifespan
async def app_lifespan(server):
    """Life span for server"""
    logger.info("Server starting up")

    # add more sessions here as required
    neuromldb_session = aiohttp.ClientSession()
    try:
        yield {"neuromldb_session": neuromldb_session}
    finally:
        logger.info("Shutting down")
        await neuromldb_session.close()
