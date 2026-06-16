#!/usr/bin/env python3
"""
Tests for tool utility functions.

File: utils_pkg/tests/test_tools_utils.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

from fastmcp.client.client import CallToolResult
from mcp.types import EmbeddedResource, TextContent, TextResourceContents
from pydantic.networks import AnyUrl

from klea_utils.tools import textualize_tool_results

logging.basicConfig(
    format="%(name)s (%(levelname)s) >>> %(message)s\n", level=logging.WARNING
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def test_textualize_tool_results_success():
    """A tool returning a dict with model info and a downloaded resource."""
    result = CallToolResult(
        content=[
            TextContent(
                type="text",
                text=(
                    '{"Model_42": {"Model_ID": "42", '
                    '"Name": "Purkinje cell", '
                    '"resource": "/home/user/.cache/nml_mcp/42.xml"}}'
                ),
            )
        ],
        structured_content={
            "Model_42": {
                "Model_ID": "42",
                "Name": "Purkinje cell",
                "resource": "/home/user/.cache/nml_mcp/42.xml",
            }
        },
        meta=None,
        data={
            "Model_42": {
                "Model_ID": "42",
                "Name": "Purkinje cell",
                "resource": "/home/user/.cache/nml_mcp/42.xml",
            }
        },
        is_error=False,
    )
    output = textualize_tool_results([result])
    logger.debug(f"{output = }")

    assert "## Tool Results" in output
    assert "Purkinje cell" in output
    assert "/home/user/.cache/nml_mcp/42.xml" in output
    assert "Error" not in output


def test_textualize_tool_results_error():
    """A tool that errored, returning an error message."""
    result = CallToolResult(
        content=[
            TextContent(
                type="text",
                text="Connection timeout: NeuroML-DB not reachable",
            )
        ],
        structured_content=None,
        meta=None,
        data=None,
        is_error=True,
    )
    output = textualize_tool_results([result])
    logger.debug(f"{output = }")

    assert "## Tool Results" in output
    assert "**Error:**" in output
    assert "Connection timeout" in output


def test_textualize_tool_results_multiple_blocks():
    """A single result with both text content and an embedded resource."""
    embedded = EmbeddedResource(
        type="resource",
        resource=TextResourceContents(
            uri=AnyUrl("file:///cache/42.xml"),
            mimeType="application/xml",
            text="<cell>Purkinje</cell>",
        ),
    )
    result = CallToolResult(
        content=[
            TextContent(type="text", text='{"status": "ok"}'),
            embedded,
        ],
        structured_content=None,
        meta=None,
        data=None,
        is_error=False,
    )
    output = textualize_tool_results([result])
    logger.debug(f"{output = }")

    assert "## Tool Results" in output
    assert '{"status": "ok"}' in output
    assert "file:///cache/42.xml" in output
    assert "<cell>Purkinje</cell>" in output


def test_textualize_tool_results_multiple_results():
    """Multiple results: one success, one error."""
    success = CallToolResult(
        content=[
            TextContent(type="text", text='{"model": "cerebellum"}'),
        ],
        structured_content=None,
        meta=None,
        data=None,
        is_error=False,
    )
    error = CallToolResult(
        content=[
            TextContent(type="text", text="Timeout fetching data"),
        ],
        structured_content=None,
        meta=None,
        data=None,
        is_error=True,
    )
    output = textualize_tool_results([success, error])
    logger.debug(f"{output = }")

    assert "## Tool Results" in output
    assert "### Result 1/2" in output
    assert "### Result 2/2" in output
    assert '{"model": "cerebellum"}' in output
    assert "**Error:**" in output
    assert "Timeout fetching data" in output
