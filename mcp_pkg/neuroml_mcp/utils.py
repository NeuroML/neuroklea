#!/usr/bin/env python3
"""
MCP utils

File: neuroml_mcp/utils.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import inspect
import shutil
from pathlib import Path
from typing import Any

from platformdirs import PlatformDirs
from pydantic import BaseModel

MCP_DIRS = PlatformDirs("nml_mcp")


def init_cache_dir():
    """Initialise cache directory if it doesn't exist."""
    cache_dir = Path(MCP_DIRS.user_cache_dir)
    # Create cache directory if it doesn't exist (parents should already exist)
    cache_dir.mkdir(parents=False, exist_ok=True)


def cleanup_cache_dir():
    """Clean up the cache contents.

    To be used at end of each session.
    """
    cache_dir = Path(MCP_DIRS.user_cache_dir)
    # Remove all contents but keep the directory
    for item in cache_dir.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


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
