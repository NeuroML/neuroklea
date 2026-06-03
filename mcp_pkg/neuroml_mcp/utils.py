#!/usr/bin/env python3
"""
MCP utils

File: neuroml_mcp/utils.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import inspect
from typing import Any

from platformdirs import PlatformDirs
from pydantic import BaseModel

MCP_DIRS = PlatformDirs("nml_mcp")


class ToolInfo(BaseModel):
    """Additional metadata"""

    description: str | None = None
    tags: set[str] | None = None
    meta: dict[str, Any] | None = None


def register_tools(mcp, modules: list):
    """Register tools from a given module

    :param modules: list of modules with tool function definitions

    """
    for module in modules:
        for fname, fn in inspect.getmembers(module, inspect.isfunction):
            if fname.endswith("_tool"):
                if hasattr(fn, "_tool_meta"):
                    metadata: ToolInfo = fn._tool_meta

                    kwargs: dict[str, Any] = {}

                    kwargs["description"] = metadata.description or fn.__doc__
                    if metadata.tags is not None:
                        kwargs["tags"] = metadata.tags
                    if metadata.meta is not None:
                        kwargs["meta"] = metadata.meta

                    mcp.tool(fn, **kwargs)
                else:
                    raise ValueError(f"{fname} is missing ToolInfo")


def tool_meta(metadata: ToolInfo):
    """Attach metadata to tools."""

    def wrapper(fn):
        fn._tool_meta = metadata
        return fn

    return wrapper
