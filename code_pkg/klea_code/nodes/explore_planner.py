#!/usr/bin/env python3
"""
Explore planner node for KleaCode

File: code_pkg/klea_code/nodes/explore_planner.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from typing import Any, Dict, override

from pydantic import BaseModel

from klea_code.nodes.planner import Planner
from klea_code.schemas import KleaCodeState, PlanSchema


class ExplorePlanner(Planner):
    """Node that plans exploration steps for a codebase.

    Subclasses Planner with:
    - Updates state.exploration_plan instead of state.task_plan
    """

    @override
    def __init__(self, logger, label: str, model, temperature: float = 0.01):
        """Initialise the explore planner node.

        :param logger: Logger instance
        :param label: Human-readable label for UI progress display
        :param model: LLM model instance
        :param temperature: Sampling temperature
        """
        super().__init__(
            logger=logger, label=label, model=model, temperature=temperature
        )
        self.prompt_prefix = "ExplorePlanner"

    @override
    def _get_prompt_variables(self, state: KleaCodeState) -> dict:
        """Format prompt with current state."""
        # TODO: limit to required state field
        return {
            "query": state.query,
            "goal": state.goal,
            "step_list": state.plan.step_list,
            "current_step_index": state.plan.current_step_index,
            "discovery": state.discovery_persistent,
            "discovery_last_step": state.discovery_per_step,
            "observations": state.step_outputs,
        }

    @override
    def _update_state(self, result: PlanSchema, state: BaseModel) -> Dict[str, Any]:
        """Update exploration_plan in state."""
        return {"plan": result}

    @override
    def _get_default_error_result(self) -> PlanSchema:
        """Return default result when processing fails."""
        return PlanSchema(status="failed")
