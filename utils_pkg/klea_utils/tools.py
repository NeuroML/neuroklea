#!/usr/bin/env python3
"""
Tool-related utilities for Klea

File: klea_utils/tools.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, List

from fastmcp.client.client import CallToolResult
from mcp.types import EmbeddedResource, TextContent

logging.basicConfig(
    format="%(name)s (%(levelname)s) >>> %(message)s\n", level=logging.WARNING
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def _textualize_content_block(block: Any) -> str:
    """Extract text from a single ContentBlock."""
    if isinstance(block, TextContent):
        return block.text
    elif isinstance(block, EmbeddedResource):
        resource = block.resource
        if hasattr(resource, "blob") and resource.blob:
            logger.warning("Blob resource not processed: %s", resource.uri)
        text = getattr(resource, "text", None) or getattr(resource, "blob", "")
        return f"[Resource: {resource.uri}]\n{text}"
    else:
        logger.warning("Unhandled content block type: %s", type(block).__name__)
        return str(block)


def textualize_tool_results(
    tool_results: List[CallToolResult],
) -> str:
    """Format tool call results as LLM-ready text for use in prompt context.

    Tool return values are typically JSON-structured data (dicts, lists) which
    LLMs handle naturally. We wrap them in markdown code blocks for clear
    separation from surrounding prompt text.

    :param tool_results: List of tool call results
    :returns: Formatted string suitable for inclusion in an LLM prompt
    """
    if not tool_results:
        return ""

    text = "## Tool Results\n"
    for i, result in enumerate(tool_results, 1):
        text += f"\n### Result {i}/{len(tool_results)}\n"

        if result.is_error:
            parts = [_textualize_content_block(c) for c in result.content]
            text += "**Error:** " + "\n".join(parts) + "\n"
        else:
            parts = [_textualize_content_block(c) for c in result.content]
            text += "```\n" + "\n".join(parts) + "\n```\n"

    return text
