#!/usr/bin/env python3
"""
Tools picker node for RAG

File: rag_pkg/gen_rag/nodes/tools_picker.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, Dict, override

from neuroml_ai_utils.nodes.base import BaseLLMNode

from gen_rag.schemas import RAGState, ToolCallsSchema


class ToolsPicker(BaseLLMNode[RAGState]):
    """Node that selects tools to augment vector store retrieval."""

    def __init__(
        self,
        logger: logging.Logger,
        model: Any,
        temperature: float = 0.01,
        tools_description: str | None = None,
    ):
        """Initialise the tools picker node.

        :param logger: Logger instance
        :param model: LLM model instance
        :param temperature: Sampling temperature
        """
        super().__init__(
            logger=logger,
            model=model,
            temperature=temperature,
            output_schema=ToolCallsSchema,
            memory=False,
        )
        self._tools_description = tools_description

    @override
    def _pre_exec(self, state: RAGState) -> bool:
        """Pre-execution check.

        If no tools description is set, no tools are available, and we skip the
        node.

        """
        if not self._tools_description:
            return False
        return True

    @override
    def _get_human_prompt(self, state: RAGState) -> str:
        """Return empty string -- this node only uses a system prompt."""
        return ""

    @override
    def _get_prompt_variables(self, state: RAGState) -> dict:
        """Format prompt with query and retrieval context."""
        return {
            "query": state.query,
            "tools_description": self._tools_description,
        }

    @override
    def _update_state(self, result: ToolCallsSchema, state: RAGState) -> Dict[str, Any]:
        """Update state with selected tool calls."""
        return {"tool_calls": result.tool_calls}

    @override
    def _get_default_error_result(self) -> ToolCallsSchema:
        """Return default result when processing fails."""
        return ToolCallsSchema()
