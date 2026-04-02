#!/usr/bin/env python3
"""
Planner node for CodeAI

File: code_ai_pkg/neuroml_code_ai/nodes/planner.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, Dict, override

from neuroml_ai_utils.nodes.base import BaseLLMNode
from pydantic import BaseModel

from neuroml_code_ai.schemas import CodeAIState, PlanSchema


class Planner(BaseLLMNode[PlanSchema]):
    """Node that creates or updates an execution plan."""

    def __init__(self, logger: logging.Logger, model: Any, temperature: float = 0.01):
        """Initialise the planner node.

        :param logger: Logger instance
        :param model: LLM model instance (reasoning model)
        :param temperature: Sampling temperature
        """
        super().__init__(
            logger=logger,
            model=model,
            temperature=temperature,
            output_schema=PlanSchema,
            memory=False,
        )
        self._tools_description = ""

    def set_tools_description(self, description: str) -> None:
        """Set tool descriptions (called by orchestrator after construction)."""
        self._tools_description = description

    @override
    def _get_prompt_variables(self, state: CodeAIState) -> dict:
        """Format prompt with current plan state."""
        return {
            "query": state.query,
            "goal": state.goal,
            "step_list": state.plan.step_list,
            "current_step_index": state.plan.current_step_index,
            "artefacts": state.artefacts,
            "observations": state.tool_responses,
            "tools_description": self._tools_description,
            "output_schema": self._get_output_schema_json(),
        }

    @override
    def _update_state(self, result: PlanSchema, state: BaseModel) -> Dict[str, Any]:
        """Update plan and generate summary for user."""
        plan_summary = "## Plan summary:\n\n"
        for step in result.step_list:
            plan_summary += f"- {step.step_number}: {step.description}"

        plan = state.plan  # type: ignore
        plan.step_list = result.step_list

        return {"plan": plan, "message_for_user": plan_summary}

    @override
    def _get_default_error_result(self) -> PlanSchema:
        """Return default result when processing fails."""
        return PlanSchema(status="failed")
