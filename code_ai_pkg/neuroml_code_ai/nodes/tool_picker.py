#!/usr/bin/env python3
"""
Tool picker node

File: code_ai_pkg/neuroml_code_ai/nodes/tool_picker.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, Dict

from langchain_core.utils.function_calling import convert_to_json_schema
from neuroml_ai_utils.nodes.base_nodes import BaseMemoryLLMNode
from pydantic import BaseModel

from neuroml_code_ai.schemas import CodeAIState, ToolCallSchema


class ToolPicker(BaseMemoryLLMNode[ToolCallSchema]):
    """Node that selects the best tool for the current step."""

    def __init__(self, logger: logging.Logger, model: Any, temperature: float = 0.01):
        """Initialise the tool picker node.

        :param logger: Logger instance
        :param model: LLM model instance (reasoning model)
        :param temperature: Sampling temperature
        """
        super().__init__(
            logger=logger,
            model=model,
            temperature=temperature,
            output_schema=ToolCallSchema,
            memory=False,
        )
        self._tools_description = ""

    def set_tools_description(self, description: str) -> None:
        """Set tool descriptions (called by orchestrator after construction)."""
        self._tools_description = description

    def _get_human_prompt(self, state: BaseModel) -> str:
        """Return empty string — this node only uses a system prompt."""
        return ""

    def _get_prompt_variables(self, state: CodeAIState) -> dict:
        """Format prompt with current step state."""
        current_step_index = state.plan.current_step_index
        current_step = state.plan.step_list[current_step_index]

        return {
            "current_step": current_step,
            "artefacts": state.artefacts,
            "observations": state.tool_responses,
            "tools_description": self._tools_description,
            "output_schema": convert_to_json_schema(ToolCallSchema),
        }

    def _update_state(self, result: ToolCallSchema, state: BaseModel) -> Dict[str, Any]:
        """Update state with the selected tool call."""
        return {"tool_call": result}

    def _get_default_error_result(self) -> ToolCallSchema:
        """Return default result when processing fails."""
        return ToolCallSchema(tool="INVALID")
