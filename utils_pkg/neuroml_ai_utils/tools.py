#!/usr/bin/env python3
"""
Tool-related utilities for NeuroML AI

File: neuroml_ai_utils/tools.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from typing import Any, Dict, List

from fastmcp.client.client import CallToolResult


def serialize_tool_results(
    tool_results: Dict[str, List[CallToolResult]],
) -> str:
    """Serialize tool call results into text for use in prompt context.

    :param tool_results: Dict mapping domain/query to list of tool call results
    :returns: Formatted string representation of tool results
    """
    serialized = ""
    for domain, results in tool_results.items():
        if results:
            serialized += f"## {domain}\n"
            for i, result in enumerate(results, 1):
                serialized += f"\n### Result {i}/{len(results)}\n"
                if hasattr(result, "is_error") and result.is_error:
                    serialized += f"Error: {result.content}\n"
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
