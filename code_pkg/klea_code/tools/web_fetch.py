#!/usr/bin/env python3
"""
Web fetch tool for bundled Klea Code tools.

File: code_pkg/klea_code/tools/web_fetch.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from typing import Annotated, Any
from urllib.parse import urlparse

import aiohttp
from bs4 import BeautifulSoup
from fastmcp import Context
from pydantic import Field

from klea_code.tools.utils import ToolInfo, tool_meta


def _html_to_text(html: str) -> str:
    """Strip HTML to plain text suitable for an LLM."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


@tool_meta(ToolInfo(tags={"bundled", "web"}))
async def web_fetch(
    ctx: Context,
    url: Annotated[str, Field(min_length=1)],
    timeout: Annotated[float, Field(ge=1.0, le=120.0)] = 30.0,
    max_chars: Annotated[int, Field(ge=1, le=1_000_000)] = 100_000,
) -> dict[str, Any]:
    """Fetch a URL and return its text content.
    Use this tool to read web pages, docs, or other HTTP resources.

    Args:
        url: HTTP or HTTPS URL to fetch.
        timeout: Request timeout in seconds.
        max_chars: Maximum number of characters of content to return.

    Returns:
        Dictionary with url, status_code, content_type, content, truncated, error.

    Example:
        web_fetch(url="https://example.com")
    """
    truncated = False
    error = ""
    content = ""
    status_code: int | None = None
    content_type = ""

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return {
            "url": url,
            "status_code": None,
            "content_type": "",
            "content": "",
            "truncated": False,
            "error": "URL must be an absolute http:// or https:// URL.",
        }

    session: aiohttp.ClientSession | None = ctx.lifespan_context.get("aiohttp_session")
    if session is None:
        return {
            "url": url,
            "status_code": None,
            "content_type": "",
            "content": "",
            "truncated": False,
            "error": "HTTP session not initialized",
        }

    # TODO: consider a standard Firefox/Chrome User-Agent; many sites block
    # custom agents and otherwise return empty or challenge pages.
    headers = {
        "Accept": "text/markdown, text/html, text/plain;q=0.9, */*;q=0.1",
        "User-Agent": "klea-code-web-fetch/0.0.1",
    }

    try:
        async with session.get(
            url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout),
            allow_redirects=True,
        ) as response:
            status_code = response.status
            content_type = response.headers.get("Content-Type", "")
            raw = await response.text()

            if "html" in content_type.lower():
                content = _html_to_text(raw)
            else:
                content = raw

            if len(content) > max_chars:
                content = content[:max_chars]
                truncated = True

            if status_code >= 400:
                error = f"HTTP {status_code}"
    except TimeoutError:
        error = f"Request timed out after {timeout} seconds."
    except aiohttp.ClientError as e:
        error = e.__str__()

    return {
        "url": url,
        "status_code": status_code,
        "content_type": content_type,
        "content": content,
        "truncated": truncated,
        "error": error,
    }
