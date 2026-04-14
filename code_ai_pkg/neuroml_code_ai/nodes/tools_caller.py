#!/usr/bin/env python3
"""
Tools caller node

File: code_ai_pkg/neuroml_code_ai/nodes/tools_caller.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, override

from fastmcp import Client
from fastmcp.client.client import CallToolResult
from neuroml_ai_utils.nodes.abstract import AbstractLangGraphNode

from neuroml_code_ai.schemas import CodeAIState


class ToolsCaller(AbstractLangGraphNode[CodeAIState, CallToolResult]):
    """Node that calls the selected tools."""

    def __init__(self, logger: logging.Logger, mcp_client: Client):
        """Initialise the tools caller node.

        :param logger: Logger instance
        :param mpc_client: MCP client instance
        """
        super().__init__(
            logger=logger,
        )
        self._mcp_client = mcp_client

    @override
    async def execute(self, state: CodeAIState) -> dict[str, Any]:
        self.logger.debug(f"{state =}")
        result: dict[str, Any] = {}

        plan = state.plan
        current_step = plan.step_list[plan.current_step_index]
        tool_responses = state.tool_responses

        # call tool if it is in the current state
        if state.tool_call:
            # TODO: retry X times if fails before marking as failed
            tool_call = state.tool_call

            async with self.mcp_client:
                tool_result = await self.mcp_client.call_tool(
                    name=tool_call.tool, arguments=tool_call.args, raise_on_error=False
                )
        else:
            return {}

        tool_responses.append(tool_result)

        if tool_result.is_error:
            current_step.status = "failed"
        else:
            current_step.status = "done"

        # TODO: populate artefacts
        result["tool_responses"] = tool_responses
        self.logger.debug(f"{tool_responses =}")

        plan.current_step_index += 1

        result["plan"] = plan
        return result
