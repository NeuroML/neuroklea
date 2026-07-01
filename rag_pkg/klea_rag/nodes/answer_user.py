#!/usr/bin/env python3
"""
Answer user node

File: rag_pkg/klea_rag/nodes/answer_user.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, Dict, override

from klea_utils.nodes.abstract import AbstractLangGraphNode

from klea_rag.schemas import RAGState


class AnswerUser(AbstractLangGraphNode[RAGState, Dict[str, Any]]):
    """Node that returns the final message to the user."""

    def __init__(self, logger: logging.Logger, label: str):
        """Initialise with a logger.

        :param logger: Logger instance
        :param label: Human-readable label for UI progress display
        """
        super().__init__(logger, label)

    @override
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
