#!/usr/bin/env python3
"""
Answer user node

File: code_ai_pkg/neuroml_code_ai/nodes/answer_user.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, Dict

from neuroml_ai_utils.nodes.base_nodes import BaseLangGraphNode

from neuroml_code_ai.schemas import CodeAIState


class AnswerUserNode(BaseLangGraphNode[CodeAIState, Dict[str, Any]]):
    """Node that returns the final message to the user."""

    def __init__(self, logger: logging.Logger):
        """Initialise with a logger.

        :param logger: Logger instance
        """
        super().__init__(logger)

    async def execute(self, state: CodeAIState) -> Dict[str, Any]:
        """Return the message for the user.

        :param state: Current graph state
        :returns: State update with message_for_user
        """
        self.logger.debug(f"{state =}")

        answer = state.message_for_user
        self.logger.info(f"Returning final answer to user: {answer}")

        return {"message_for_user": answer}
