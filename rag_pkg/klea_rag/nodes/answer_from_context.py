#!/usr/bin/env python3
"""
Generate an answer from provided reference material

File: rag_pkg/klea_rag/nodes/answer_from_context.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, Dict, override

from klea_utils.llm import split_output_by_section
from klea_utils.nodes.base import BaseLLMNode
from klea_utils.stores import serialize_vs_retrieval
from klea_utils.tools import textualize_tool_results
from langchain.messages import AIMessage
from pydantic import BaseModel, Field


class AnswerSchema(BaseModel):
    answer: str = ""
    references: list[str] = Field(default_factory=list)


class AnswerFromContext(BaseLLMNode[AnswerSchema]):
    """Generate an answer from the provided context"""

    def __init__(
        self,
        logger: logging.Logger,
        model: Any,
        temperature: float = 0.3,
        memory: bool = False,
    ):
        """Initialise the node.

        :param logger: Logger instance
        :param model: LLM model instance
        :param temperature: Sampling temperature
        :param memory: Whether to include conversation memory in the prompt
        """
        super().__init__(
            logger=logger,
            model=model,
            temperature=temperature,
            output_schema=AnswerSchema,
            memory=memory,
        )

    @override
    def _get_prompt_variables(self, state: BaseModel) -> dict:
        """Format prompt with question and serialized reference material."""
        reference_material = state.reference_material  # type: ignore
        reference_material_text = serialize_vs_retrieval(reference_material)

        # Add tool results to the reference material
        if hasattr(state, "tool_results") and state.tool_results:  # type: ignore
            reference_material_text += "\n" + textualize_tool_results(
                state.tool_results
            )  # type: ignore

        return {
            "query": state.query,  # type: ignore
            "reference_material": reference_material_text,
        }

    @override
    def _update_state(self, result: AnswerSchema, state: BaseModel) -> Dict[str, Any]:
        """Update state with the generated answer and formatted references."""
        thought, answer = split_output_by_section(result.answer, "<think>", "</think>")
        refs = result.references

        full_answer = self._update_reference_list(answer, refs)
        res_message = AIMessage(content=full_answer)
        self.logger.debug(res_message.pretty_repr())

        messages = state.messages  # type: ignore
        messages.append(res_message)

        return {
            "messages": messages,
            "reference_material": state.reference_material,  # type: ignore
        }

    def _update_reference_list(self, answer: str, references: list[str]) -> str:
        """Update answer with reference list

        Override with ``pass`` to skip reference listing entirely,
        or override with custom formatting logic.

        We rely on the LLM to generate an output with a reference list, since
        we want it to only list references that it used in the answer.

        :param answer: The answer returned from the LLM, with references
        :returns: answer text with formatted references if available
        """
        full_answer = f"{answer}"
        newrefs = list(set([r.strip() for r in references]))

        if len(newrefs):
            full_answer += "\n\nReferences:\n"
            for r in newrefs:
                full_answer += f"- {r}\n"
        else:
            self.logger.debug("No references included.")

        return full_answer

    @override
    def _get_default_error_result(self) -> Any:
        """Return default result when processing fails."""
        return ""
