#!/usr/bin/env python3
"""
Tools caller node for RAG

File: rag_pkg/klea_rag/nodes/tools_caller.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import asyncio
import logging
from typing import Any, Dict, override

from fastmcp import Client
from fastmcp.client.client import CallToolResult
from klea_utils.nodes.abstract import AbstractLangGraphNode

from klea_rag.schemas import RAGState


class ToolsCaller(AbstractLangGraphNode[RAGState, Dict[str, Any]]):
    """Node that calls MCP tools based on tool_calls in state."""

    def __init__(self, logger: logging.Logger, mcp_client: Client):
        """Initialise the tools caller node.

        :param logger: Logger instance
        :param mcp_client: MCP client instance
        """
        super().__init__(logger=logger)
        self._mcp_client = mcp_client

    @override
    async def execute(self, state: RAGState) -> Dict[str, Any]:
        self.logger.debug(f"{state =}")

        # no _pre_exec here
        if not state.tool_calls or not self._mcp_client:
            self.logger.debug("Pre-exec check failed, skipping execution")
            return {}

        results: list[CallToolResult] = []

        async with self._mcp_client:
            tasks = [
                self._mcp_client.call_tool(
                    name=tc.tool,
                    arguments=tc.args,
                    raise_on_error=False,
                )
                for tc in state.tool_calls
            ]
            results = await asyncio.gather(*tasks)

        self.logger.debug(f"{results =}")
        # Replace because we want fresh results at RAG loop
        return {"tool_results": results}
