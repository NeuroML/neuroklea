#!/usr/bin/env python3
"""
Initialise RAG state node

File: rag_pkg/klea_rag/nodes/init_rag.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, Dict

from klea_utils.nodes.abstract import AbstractLangGraphNode

from klea_rag.schemas import EvaluateAnswerSchema, RAGState


class InitRAGState(AbstractLangGraphNode[RAGState, Dict[str, Any]]):
    """Initialise/reset RAG state before each iteration."""

    def __init__(self, logger: logging.Logger):
        """Initialise with a logger."""
        super().__init__(logger)

    async def execute(self, state: RAGState) -> Dict[str, Any]:
        """Reset state fields to their initial values."""
        return {
            "guard_decision": "unsafe",
            "text_response_eval": EvaluateAnswerSchema(),
            "message_for_user": "",
            "retrieval_attempts": 0,
            "rewrite_attempts": 0,
            "retrieval_query": "",
            "tool_calls": [],
            "tool_results": [],
            "reference_material": {},
        }
