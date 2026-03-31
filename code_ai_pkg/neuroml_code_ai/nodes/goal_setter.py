#!/usr/bin/env python3
"""
Goal setter node

File: code_ai_pkg/neuroml_code_ai/nodes/goal_setter.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from pathlib import Path
from typing import Any, Dict

from neuroml_ai_utils.nodes.base_nodes import BaseMemoryLLMNode
from pydantic import BaseModel

from neuroml_code_ai import prompts
from neuroml_code_ai.schemas import CodeAIState, GoalSchema


class GoalSetterNode(BaseMemoryLLMNode[GoalSchema]):
    """Goal setter node"""

    def __init__(
        self,
        logger,
        model,
        temperature,
        output_schema,
        system_prompt_file,
        human_prompt_file,
        memory,
    ):
        super().__init__(
            logger,
            model,
            temperature,
            output_schema,
            system_prompt_file,
            human_prompt_file,
            prompt_registry_location=Path(prompts.__file__).parent,
            memory=memory,
        )

    def _get_prompt_variables(self, state: CodeAIState) -> dict:
        """Format prompt with state-specific parameters"""
        variables = {"query": state.query, "context_summary": ""}
        self.logger.debug(f"{variables =}")
        return variables

    def _update_state(self, result: GoalSchema, state: BaseModel) -> Dict[str, Any]:
        """Update and return state dictionary"""
        state_update = {"goal": result, "message_for_user": result.goal}
        self.logger.debug(state_update)
        return state_update

    def _get_default_error_result(self) -> GoalSchema:
        """Return default result when processing fails"""
        return self._get_output_schema()(goal="Invalid", success_criteria="Invalid")
