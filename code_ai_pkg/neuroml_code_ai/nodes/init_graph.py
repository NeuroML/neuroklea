#!/usr/bin/env python3
"""
Initialise graph state node

File: code_ai_pkg/neuroml_code_ai/nodes/init_graph.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, Dict, override

from neuroml_ai_utils.nodes.base_nodes import BaseLangGraphNode

from neuroml_code_ai.schemas import CodeAIState, GoalSchema, PlanSchema


class InitGraphState(BaseLangGraphNode[CodeAIState, Dict[str, Any]]):
    """Initialise/reset graph state before each iteration."""

    def __init__(self, logger: logging.Logger):
        """Initialise with a logger."""
        super().__init__(logger)

    @override
    async def execute(self, state: CodeAIState) -> Dict[str, Any]:
        """Reset state fields to their initial values."""
        return {
            "message_for_user": "",
            "plan": PlanSchema(),
            "goal": GoalSchema(),
            "tool_call": None,
            "tool_responses": [],
        }
