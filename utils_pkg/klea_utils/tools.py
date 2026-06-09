#!/usr/bin/env python3
"""
Tool-related utilities for NeuroML AI

File: klea_utils/tools.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from typing import List

from fastmcp.client.client import CallToolResult


def serialize_tool_results(
    tool_results: List[CallToolResult],
) -> str:
    """Serialize tool call results into text for use in prompt context.

    :param tool_results: List of tool call results
    :returns: Formatted string representation of tool results
    """
    serialized = ""
    if tool_results:
        serialized += "## Tool Results\n"
        for i, result in enumerate(tool_results, 1):
            serialized += f"\n### Result {i}/{len(tool_results)}\n"
            if hasattr(result, "is_error") and result.is_error:
                pass
            elif hasattr(result, "content"):
                # Extract text content from ContentBlock objects
                text_content = []
                for content_block in result.content:
                    if hasattr(content_block, "text"):
                        text_content.append(content_block.text)
                    elif hasattr(content_block, "__str__"):
                        text_content.append(str(content_block))
                serialized += "\n".join(text_content) + "\n"
            else:
                serialized += str(result) + "\n"

    # Escape braces for template formatting
    serialized = serialized.replace("{", "{{").replace("}", "}}")
    return serialized
