#!/usr/bin/env python3
"""
Evaluator node for RAG

File: rag_pkg/gen_rag/nodes/evaluator.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, Dict

from langchain_core.utils.function_calling import convert_to_json_schema
from neuroml_ai_utils.nodes.base_nodes import BaseMemoryLLMNode
from neuroml_ai_utils.stores import serialize_reference
from pydantic import BaseModel

from gen_rag.schemas import EvaluateAnswerSchema, RAGState


class Evaluator(BaseMemoryLLMNode[EvaluateAnswerSchema]):
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

    def _get_prompt_variables(self, state: RAGState) -> dict:
        """Format prompt with question, context, and answer."""
        question = state.query
        context = serialize_reference(state.reference_material)
        answer = state.messages[-1].content

        return {
            "question": question,
            "context": context,
            "answer": answer,
            "output_schema": self._get_output_schema_json(),
        }

    def _update_state(
        self, result: EvaluateAnswerSchema, state: BaseModel
    ) -> Dict[str, Any]:
        """Update state with evaluation result."""
        return {"text_response_eval": result, "messages": state.messages}

    def _get_default_error_result(self) -> EvaluateAnswerSchema:
        """Return default result when processing fails."""
        return EvaluateAnswerSchema(next_step="undefined", summary="Evaluation failed")

    def _get_output_schema_json(self) -> str:
        """Get JSON schema string for the prompt."""
        return convert_to_json_schema(self.output_schema)
