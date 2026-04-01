#!/usr/bin/env python3
"""
Answer user node

File: rag_pkg/gen_rag/nodes/answer_user.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, Dict

from neuroml_ai_utils.nodes.base_nodes import BaseLangGraphNode

from gen_rag.schemas import RAGState


class AnswerUser(BaseLangGraphNode[RAGState, Dict[str, Any]]):
    """Node that returns the final message to the user."""

    def __init__(self, logger: logging.Logger):
        """Initialise with a logger.

        :param logger: Logger instance
        """
        super().__init__(logger)

    async def execute(self, state: RAGState) -> Dict[str, Any]:
        """Return the message for the user.

        :param state: Current graph state
        :returns: State update with message_for_user
        """
        self.logger.debug(f"{state =}")

        messages = state.messages
        answer = messages[-1]

        self.logger.info(f"Returning final answer to user: {answer}")

        return {"message_for_user": answer.content}
