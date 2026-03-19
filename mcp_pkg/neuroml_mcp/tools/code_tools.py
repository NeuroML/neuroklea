#!/usr/bin/env python3
"""
General code execution tools.

Note that docstrings here should be written for the LLM to read.

File: neuroml_mcp/tools/code_tools.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field
from typing_extensions import Annotated

from neuroml_mcp.tools.sandbox.sandbox import RunPythonCode

from .sandbox import nml_mcp_sandbox

# set the implementation for development
sbox = nml_mcp_sandbox


async def dummy_code_tool(
    astring: Annotated[str, Field(description="String to be echoed back")],
) -> str:
    """Return the input string in a sentence (testing tool only).

    This is a dummy tool used only for unit testing and debugging.

    Example: dummy_code_tool("hello") returns "I got hello"
    """
    return f"I got {astring}"


async def list_files_tool(
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
        Optional[int],
        Field(description="Maximum directory depth to traverse. 'None' for unlimited"),
    ] = None,
    # LLMs are trained on shell style globs, so they insist on using space
    # separated file patterns. So we explicitly support these. Otherwise, this
    # becomes error prone.
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
) -> Dict[str, Any]:
    """List files and directories with filtering and metadata.
    Use this tool to explore file system structure and find specific files.

    Example: list_files_tool(path=".", pattern="*.py", recursive=True)
    """
    the_path = Path(path)
    truncated = "False"
    error = ""
    files: List[Dict[str, Any]] = []
    paths: List[Path] = []

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


async def run_python_code_tool(
    code: Annotated[
        str,
        Field(
            description=(
                "Complete Python code to execute. Must be valid Python syntax "
                "and cannot require interactive input"
            ),
            min_length=1,
        ),
    ],
) -> Dict[str, Any]:
    """Execute Python code in a sandboxed environment.

    Use this tool to test code snippets, generate models, and perform calculations.

    Example: run_python_code_tool(
        "import numpy; print('numpy version:', numpy.__version__)"
    )
    """
    request = RunPythonCode(code=code)
    async with sbox(".") as f:
        result = await f.run(request)
    return asdict(result)
