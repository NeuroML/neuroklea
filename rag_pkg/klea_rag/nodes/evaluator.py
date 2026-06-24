#!/usr/bin/env python3
"""
Evaluator node for RAG

File: rag_pkg/klea_rag/nodes/evaluator.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, Dict, override

from klea_utils.nodes.base import BaseLLMNode
from klea_utils.stores.utils import serialize_vs_retrieval

from klea_rag.schemas import EvaluateAnswerSchema, RAGState


class Evaluator(BaseLLMNode[EvaluateAnswerSchema]):
    """Node that evaluates a RAG-generated answer against retrieved context."""

    def __init__(self, logger: logging.Logger, model: Any, temperature: float = 0.0):
        """Initialise the evaluator node.

        :param logger: Logger instance
        :param model: LLM model instance
        :param temperature: Sampling temperature
        """
        super().__init__(
            logger=logger,
            model=model,
            temperature=temperature,
            output_schema=EvaluateAnswerSchema,
            memory=False,
        )

    @override
    def _get_prompt_variables(self, state: RAGState) -> dict:
        """Format prompt with question, context, and answer."""
        question = state.query
        context = serialize_vs_retrieval(state.reference_material)
        answer = state.messages[-1].content

        return {
            "question": question,
            "context": context,
            "answer": answer,
        }

    @override
    def _update_state(
        self, result: EvaluateAnswerSchema, state: RAGState
    ) -> Dict[str, Any]:
        """Update state with evaluation result and computed routing decision."""
        return {
            "text_response_eval": result,
        }

    @override
    def _get_default_error_result(self) -> EvaluateAnswerSchema:
        """Return default result when processing fails."""
        return EvaluateAnswerSchema(next_step="undefined", summary="Evaluation failed")
