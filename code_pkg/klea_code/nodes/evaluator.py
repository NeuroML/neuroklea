#!/usr/bin/env python3
"""
Evaluator node for KleaCode

File: code_pkg/klea_code/nodes/evaluator.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

# TODO: complete

import logging
from typing import Any, Dict, override

from klea_utils.nodes.abstract import AbstractLangGraphNode

from klea_code.schemas import KleaCodeState


class Evaluator(AbstractLangGraphNode[KleaCodeState, Dict[str, Any]]):
    """Node that evaluates whether all plan steps are completed."""

    def __init__(self, logger: logging.Logger, label: str):
        """Initialise with a logger.

        :param logger: Logger instance
        :param label: Human-readable label for UI progress display
        """
        super().__init__(logger, label)

    @override
    async def execute(self, state: KleaCodeState) -> Dict[str, Any]:
        """Check if all steps are completed and update plan status.

        :param state: Current graph state
        :returns: State update with plan status
        """
        self.write_custom_stream({"type": "progress", "node": self.label})
        self.logger.debug(f"{state =}")
        plan = state.plan
        result: Dict[str, Any] = {}

        if plan.current_step_index >= len(plan.step_list):
            plan.status = "completed"
            result["plan"] = plan

        return result
