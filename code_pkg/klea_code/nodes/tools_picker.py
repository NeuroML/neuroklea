#!/usr/bin/env python3
"""
Tools picker node

File: code_pkg/klea_code/nodes/tools_picker.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, Dict, override

from klea_utils.nodes.base import BaseLLMNode

from klea_code.schemas import KleaCodeState, ToolCallSchema


class ToolsPicker(BaseLLMNode[KleaCodeState]):
    """Node that selects the best tools for the current step."""

    def __init__(
        self, logger: logging.Logger, label: str, model: Any, temperature: float = 0.01
    ):
        """Initialise the tools picker node.

        :param logger: Logger instance
        :param label: Human-readable label for UI progress display
        :param model: LLM model instance (reasoning model)
        :param temperature: Sampling temperature
        """
        super().__init__(
            logger=logger,
            label=label,
            model=model,
            temperature=temperature,
            output_schema=ToolCallSchema,
            memory=False,
        )
        self._tools_description = ""

    def set_tools_description(self, description: dict[str, str]) -> None:
        """Set tool descriptions (called by orchestrator after construction)."""
        self._tools_description = (
            "\n\n".join(description.values()) if description else ""
        )

    @override
    def _get_human_prompt(self, state: KleaCodeState) -> str:
        """Return empty string  ---  this node only uses a system prompt."""
        return ""

    @override
    def _get_prompt_variables(self, state: KleaCodeState) -> dict:
        """Format prompt with current step state."""
        current_step_index = state.plan.current_step_index
        current_step = state.plan.step_list[current_step_index]

        return {
            "current_step": current_step,
            "artefacts": state.artefacts,
            "observations": state.tool_responses,
            "tools_description": self._tools_description,
        }

    @override
    def _update_state(
        self, result: ToolCallSchema, state: KleaCodeState
    ) -> Dict[str, Any]:
        """Update state with the selected tool call."""
        return {"tool_call": result}

    @override
    def _get_default_error_result(self) -> ToolCallSchema:
        """Return default result when processing fails."""
        return ToolCallSchema(tool="INVALID")
