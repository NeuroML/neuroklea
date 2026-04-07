#!/usr/bin/env python3
"""
Goal setter node

File: code_ai_pkg/neuroml_code_ai/nodes/goal_setter.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, Dict, override

from neuroml_ai_utils.nodes.base import BaseLLMNode
from pydantic import BaseModel

from neuroml_code_ai.schemas import CodeAIState, GoalSchema


class GoalSetter(BaseLLMNode[GoalSchema]):
    """Goal setter node"""

    def __init__(
        self,
        logger: logging.Logger,
        model: Any,
        temperature: float,
        output_schema: type[GoalSchema],
        memory: bool = False,
    ):
        """Initialise the goal setter node.

        :param logger: Logger instance
        :param model: LLM model instance
        :param temperature: Sampling temperature
        :param output_schema: Pydantic schema for structured output
        :param memory: Whether to append memory content to the system prompt
        """
        super().__init__(
            logger=logger,
            model=model,
            temperature=temperature,
            output_schema=output_schema,
            memory=memory,
        )

    @override
    def _get_prompt_variables(self, state: CodeAIState) -> dict:
        """Format prompt with state-specific parameters"""
        variables = {"query": state.query}
        self.logger.debug(f"{variables =}")
        return variables

    @override
    def _update_state(self, result: GoalSchema, state: BaseModel) -> Dict[str, Any]:
        """Update and return state dictionary"""
        state_update = {"goal": result, "message_for_user": result.goal}
        self.logger.debug(state_update)
        return state_update

    @override
    def _get_default_error_result(self) -> GoalSchema:
        """Return default result when processing fails"""
        return self.output_schema(goal="Invalid", success_criteria="Invalid")
