#!/usr/bin/env python3
"""
Evaluator node for CodeAI

File: code_ai_pkg/neuroml_code_ai/nodes/evaluator.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

# TODO: complete

import logging
from typing import Any, Dict, override

from neuroml_ai_utils.nodes.abstract import AbstractLangGraphNode

from neuroml_code_ai.schemas import CodeAIState


class Evaluator(AbstractLangGraphNode[CodeAIState, Dict[str, Any]]):
    """Node that evaluates whether all plan steps are completed."""

    def __init__(self, logger: logging.Logger):
        """Initialise with a logger.

        :param logger: Logger instance
        """
        super().__init__(logger)

    @override
    async def execute(self, state: CodeAIState) -> Dict[str, Any]:
        """Check if all steps are completed and update plan status.

        :param state: Current graph state
        :returns: State update with plan status
        """
        self.logger.debug(f"{state =}")
        plan = state.plan
        result: Dict[str, Any] = {}

        if plan.current_step_index >= len(plan.step_list):
            plan.status = "completed"
            result["plan"] = plan

        return result
