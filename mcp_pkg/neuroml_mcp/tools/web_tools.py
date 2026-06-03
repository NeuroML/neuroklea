#!/usr/bin/env python3
"""
General related tools

File: mcp_pkg/neuroml_mcp/tools/web_tools.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import asyncio
import logging
import sys
from pathlib import Path

import aiohttp
from neuroml_ai_utils.plogging import (
    LoggerInfoFilter,
    LoggerNotInfoFilter,
    logger_formatter_info,
    logger_formatter_other,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from ..utils import MCP_DIRS

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.addFilter(LoggerInfoFilter())
stdout_handler.setFormatter(logger_formatter_info)
logger.addHandler(stdout_handler)

stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.DEBUG)
stderr_handler.addFilter(LoggerNotInfoFilter())
stderr_handler.setFormatter(logger_formatter_other)
logger.addHandler(stderr_handler)


@retry(
    wait=wait_random_exponential(multiplier=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    reraise=True,
)
async def _download_file_by_content(session, url, model_id, timeout, disk_file_name):
    """Download a file content and save to file"""
    r = await session.get(url, params={"modelID": model_id}, timeout=timeout, ssl=False)
    async with r:
        if r.ok:
            file_contents = await r.text()
            file_path = MCP_DIRS.user_cache_dir / Path(disk_file_name)
            with open(file_path, "w") as f:
                f.write(file_contents)
            return file_path
    return None
