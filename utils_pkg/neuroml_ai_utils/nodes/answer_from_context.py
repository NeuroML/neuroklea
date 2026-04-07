#!/usr/bin/env python3
"""
Generate an answer from provided reference material

File: utils_pkg/neuroml_ai_utils/nodes/answer_from_context.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, Dict, override

from langchain.messages import AIMessage
from pydantic import BaseModel

from neuroml_ai_utils.llm import split_output_by_section
from neuroml_ai_utils.stores import serialize_reference

from .base import BaseLLMNode


class AnswerFromContext[TSchema: BaseModel](BaseLLMNode[TSchema]):
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
            output_schema=None,
            memory=memory,
        )

    @override
    def _get_prompt_variables(self, state: BaseModel) -> dict:
        """Format prompt with question and serialized reference material."""
        reference_material = state.reference_material  # type: ignore
        reference_material_text = serialize_reference(reference_material)
        return {
            "query": state.query,  # type: ignore
            "reference_material": reference_material_text,
        }

    @override
    def _update_state(self, result: Any, state: BaseModel) -> Dict[str, Any]:
        """Update state with the generated answer and formatted references."""
        assert isinstance(result, AIMessage)
        assert isinstance(result.content, str)

        thought, answer = split_output_by_section(result.content, "<think>", "</think>")

        # Extract and deduplicate references
        answer_text = ""
        if not answer_text:
            answer_text = answer

        answer_text = self._update_reference_list(answer_text)

        result.content = answer_text
        self.logger.debug(result.pretty_repr())

        messages = state.messages  # type: ignore
        messages.append(result)

        return {
            "messages": messages,
            "reference_material": state.reference_material,  # type: ignore
        }

    def _update_reference_list(self, answer: str) -> str:
        """Update answer with reference list

        Override with ``pass`` to skip reference listing entirely,
        or override with custom formatting logic.

        We rely on the LLM to generate an output with a reference list, since
        we want it to only list references that it used in the answer.

        :param answer: The answer returned from the LLM, with references
        :returns: answer text with formatted references if available
        """
        ref_list = []
        answer_text = ""
        for rf in ["\nreference", "\nReference", "\nReferences", "\nreferences"]:
            if rf in answer:
                answer_text, references = split_output_by_section(answer, rf)
                ref_list = references.split()
                # remove whitespaces, make unique, restore in list
                ref_list = list(set(map(str.strip, ref_list)))
                break

        if ref_list:
            answer_text += "\nReferences:" + "\n- ".join(ref_list)
        else:
            self.logger.warning("No reference list found in answer.")

        return answer_text

    @override
    def _get_default_error_result(self) -> Any:
        """Return default result when processing fails."""
        return ""
