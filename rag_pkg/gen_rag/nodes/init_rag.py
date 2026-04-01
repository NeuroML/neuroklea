#!/usr/bin/env python3
"""
Initialise RAG state node

File: rag_pkg/gen_rag/nodes/init_rag.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, Dict

from neuroml_ai_utils.nodes.base_nodes import BaseLangGraphNode

from gen_rag.schemas import EvaluateAnswerSchema, RAGState


class InitRAGState(BaseLangGraphNode[RAGState, Dict[str, Any]]):
    """Initialise/reset RAG state before each iteration."""

    def __init__(self, logger: logging.Logger):
        """Initialise with a logger."""
        super().__init__(logger)

    async def execute(self, state: RAGState) -> Dict[str, Any]:
        """Reset state fields to their initial values."""
        return {
            "query_domain": "undefined",
            "text_response_eval": EvaluateAnswerSchema(),
            "message_for_user": "",
            "reference_material": {},
        }
