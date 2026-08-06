#!/usr/bin/env python3
"""
Bundled tools server for Klea Code.

Provides core tools that run in-process via stdio MCP transport,
eliminating the need for an external MCP server for common operations.

File: code_pkg/klea_code/tools/bundled.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from pathlib import Path
from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

from klea_code.tools.app_lifespan import app_lifespan
from klea_code.tools.utils import ToolInfo, register_tool, tool_meta
from klea_code.tools.web_fetch import web_fetch

bundle_server = FastMCP(
    "KleaBundled",
    instructions="Built-in tools for code exploration and file operations.",
    lifespan=app_lifespan,
)

register_tool(bundle_server, web_fetch)


@tool_meta(ToolInfo(tags={"bundled", "files"}))
async def list_files(
    path: Annotated[
        str,
        Field(
            description=(
                "Directory path to list. Must be relative to current working "
                "directory and cannot contain '..' for security"
            ),
            min_length=1,
        ),
    ],
    max_depth: Annotated[
        int | None,
        Field(description="Maximum directory depth to traverse. 'None' for unlimited"),
    ] = None,
    pattern: Annotated[
        str,
        Field(
            description=(
                """
                Space separated file patterns to filter based on files type.
                Correct: '*.py'
                Correct: '*.md'
                Correct: '*.py *.md'
            """
            )
        ),
    ] = "*",
    include_files: Annotated[
        bool, Field(description="Whether to include files in results")
    ] = True,
    include_directories: Annotated[
        bool, Field(description="Whether to include directories in results")
    ] = True,
    recursive: Annotated[
        bool, Field(description="If True, traverse subdirectories recursively")
    ] = False,
    max_results: Annotated[
        int, Field(description="Maximum number of entries to return", ge=1, le=10000)
    ] = 100,
) -> dict[str, Any]:
    """List files and directories with filtering and metadata.
    Use this tool to explore file system structure and find specific files.

    Example: list_files(path=".", pattern="*.py", recursive=True)
    """
    the_path = Path(path)
    truncated = "False"
    error = ""
    files: list[dict[str, Any]] = []
    paths: list[Path] = []

    if ".." in path:
        return {
            "files": [],
            "truncated": "False",
            "error": "Path contains '..', exiting.",
        }

    patterns = pattern.split()
    patterns = list(set(patterns))

    try:
        for p in patterns:
            if recursive:
                paths.extend(list(the_path.rglob(p)))
            else:
                paths.extend(list(the_path.glob(p)))

        if len(paths) > max_results:
            truncated = "True"

        for f in paths[:max_results]:
            ftype = "file"
            if f.is_dir():
                ftype = "directory"
            if f.is_symlink():
                ftype = "link"
            files.append(
                {
                    "path": str(f),
                    "type": ftype,
                    "modified time": f.stat().st_mtime,
                    "size": f.stat().st_size,
                }
            )
    except Exception as e:
        error = e.__str__()

    result = {"files": files, "error": error, "truncated": truncated}

    return result


register_tool(bundle_server, list_files)


if __name__ == "__main__":
    bundle_server.run(transport="stdio")
