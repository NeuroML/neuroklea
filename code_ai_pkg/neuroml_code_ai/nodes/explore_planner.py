#!/usr/bin/env python3
"""
Explore planner node for CodeAI

File: code_ai_pkg/neuroml_code_ai/nodes/explore_planner.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from typing import Any, Dict, override

from pydantic import BaseModel

from neuroml_code_ai.nodes.planner import Planner
from neuroml_code_ai.schemas import CodeAIState, PlanSchema


class ExplorePlanner(Planner):
    """Node that plans exploration steps for a codebase.

    Subclasses Planner with:
    - Updates state.exploration_plan instead of state.task_plan
    """

    @override
    def __init__(self, logger, model, temperature: float = 0.01):
        """Initialise the explore planner node.

        :param logger: Logger instance
        :param model: LLM model instance
        :param temperature: Sampling temperature
        """
        super().__init__(logger=logger, model=model, temperature=temperature)
        self.prompt_prefix = "ExplorePlanner"

    @override
    def _get_prompt_variables(self, state: CodeAIState) -> dict:
        """Format prompt with current state."""
        # TODO: limit to required state field
        return {
            "query": state.query,
            "goal": state.goal,
            "step_list": state.exploration_plan.step_list,
            "current_step_index": state.exploration_plan.current_step_index,
            "observations": state.tool_responses,
        }

    @override
    def _update_state(self, result: PlanSchema, state: BaseModel) -> Dict[str, Any]:
        """Update exploration_plan in state."""
        return {"exploration_plan": result}

    @override
    def _get_default_error_result(self) -> PlanSchema:
        """Return default result when processing fails."""
        return PlanSchema(status="failed")
