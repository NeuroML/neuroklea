#!/usr/bin/env python3
"""
Generate retrieval query node

File: rag_pkg/klea_rag/nodes/generate_retrieval_query.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from textwrap import dedent
from typing import Any, Dict, override

from langchain_core.messages import AIMessage
from langchain_core.runnables.utils import Output
from neuroml_ai_utils.nodes.base import BaseLLMNode

from klea_rag.schemas import RAGState


class GenerateRetrievalQuery(BaseLLMNode[RAGState]):
    """Node that generates a concise retrieval query from the user's question."""

    def __init__(self, logger: logging.Logger, model: Any, temperature: float = 0.3):
        """Initialise the node.

        :param logger: Logger instance
        :param model: LLM model instance
        :param temperature: Sampling temperature
        """
        super().__init__(
            logger=logger,
            model=model,
            temperature=temperature,
            output_schema=None,
            memory=True,
        )

    @override
    def _get_system_prompt(self, state: RAGState) -> str:
        """Load system prompt, optionally appending evaluator feedback."""
        system_prompt = super()._get_system_prompt(state)

        if state.query_modified:
            feedback = state.text_response_eval.summary
            system_prompt += dedent(f"""
                Generate a new query on EXACTLY one of the concepts that the
                evaluator's feedback says is missing.

                Evaluator feedback:
                {feedback}
            """)

        return system_prompt

    @override
    def _get_prompt_variables(self, state: RAGState) -> dict:
        """Format prompt with user query."""
        return {"query": state.query}

    @override
    def _update_state(self, result: Output, state: RAGState) -> Dict[str, Any]:
        """Update state with the generated retrieval query."""
        thought, answer = (
            result.content.split("</think>", 1)
            if "</think>" in result.content
            else ("", result.content)
        )
        answer = answer.strip()

        messages = state.messages
        output = AIMessage(content=answer)
        messages.append(output)

        return {"messages": messages, "retrieval_query": answer}

    @override
    def _get_default_error_result(self) -> Any:
        """Return default result when processing fails."""
        return ""
