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

from klea_utils.nodes.base import BaseLLMNode
from langchain_core.messages import AIMessage
from langchain_core.runnables.utils import Output

from klea_rag.schemas import RAGState


class GenerateRetrievalQuery(BaseLLMNode[RAGState]):
    """Node that generates a concise retrieval query from the user's question."""

    def __init__(
        self, logger: logging.Logger, label: str, model: Any, temperature: float = 0.3
    ):
        """Initialise the node.

        :param logger: Logger instance
        :param label: Human-readable label for UI progress display
        :param model: LLM model instance
        :param temperature: Sampling temperature
        """
        super().__init__(
            logger=logger,
            label=label,
            model=model,
            temperature=temperature,
            output_schema=None,
            memory=True,
        )

    @override
    def _get_system_prompt(self, state: RAGState) -> str:
        """Load system prompt, optionally appending evaluator feedback."""
        system_prompt = super()._get_system_prompt(state)

        if state.retrieval_attempts > 0:
            self.logger.info("Regenerating retrieval query, updating system prompt")
            sentence, newline, rest = system_prompt.partition("\n")
            new_sentence = dedent(
                """
                Generate a new concise retrieval query from the user's question. Think about the user's intent step by step.
                Take the evaluator's feedback into account.

                Previous query: {previous}

                Evaluator feedback:

                {feedback}

                """
            )
            system_prompt = new_sentence + rest
            self.logger.debug(f"New {system_prompt =}")

        return system_prompt

    @override
    def _get_prompt_variables(self, state: RAGState) -> dict:
        """Format prompt with user query."""
        return {
            "query": state.query,
            "feedback": state.text_response_eval.summary,
            "previous": state.retrieval_query,
        }

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

        return {
            "messages": messages,
            "retrieval_query": answer,
            "retrieval_attempts": state.retrieval_attempts + 1,
        }

    @override
    def _get_default_error_result(self) -> Any:
        """Return default result when processing fails."""
        return ""
