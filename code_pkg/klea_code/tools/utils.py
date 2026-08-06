#!/usr/bin/env python3
"""
Helpers for bundled Klea Code tools.

File: code_pkg/klea_code/tools/utils.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from typing import Any

from pydantic import BaseModel


class ToolInfo(BaseModel):
    """Additional metadata for tool registration."""

    description: str | None = None
    tags: set[str] | None = None
    meta: dict[str, Any] | None = None


def tool_meta(metadata: ToolInfo):
    """Attach metadata to tools."""

    def wrapper(fn):
        fn._tool_meta = metadata
        return fn

    return wrapper


def register_tool(mcp, fn) -> None:
    """Register a tool function that has @tool_meta metadata."""
    if not hasattr(fn, "_tool_meta"):
        raise ValueError(f"{fn.__name__} is missing ToolInfo")

    metadata: ToolInfo = fn._tool_meta
    kwargs: dict[str, Any] = {}
    kwargs["description"] = metadata.description or fn.__doc__
    if metadata.tags is not None:
        kwargs["tags"] = metadata.tags
    if metadata.meta is not None:
        kwargs["meta"] = metadata.meta

    mcp.tool(fn, **kwargs)
