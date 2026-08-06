#!/usr/bin/env python3
"""
Lifespan for the bundled Klea Code tools server.

File: code_pkg/klea_code/tools/app_lifespan.py

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
    """Create shared resources for the bundled tools server."""
    logger.info("Bundled tools server starting up")

    aiohttp_session = aiohttp.ClientSession()

    try:
        yield {"aiohttp_session": aiohttp_session}
    finally:
        logger.info("Bundled tools server shutting down")
        await aiohttp_session.close()
        logger.info("Bundled tools server shut down")
