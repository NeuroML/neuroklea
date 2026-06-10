#!/usr/bin/env python3
"""
Answer general question node

File: klea_utils/nodes/answer_general.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from pathlib import Path
from typing import Any, Dict, override

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from ..llm import split_output_by_section
from .base import BaseLLMNode


class FallbackConfig(BaseModel):
    enabled: bool = False
    warning: str = ""


class AnswerGeneral(BaseLLMNode):
    """Answer general (non-domain) questions using the LLM's training data.

    Provides a conversational, user-friendly response. Optionally appends
    conversation history for context and a fallback warning when configured.
    """

    def __init__(
        self,
        logger: logging.Logger,
        model: Any,
        temperature: float = 0.3,
        memory: bool = False,
        num_history_messages: int = 10,
        fallback_config: FallbackConfig | None = None,
    ):
        """Initialise the general answer node.

        :param logger: Logger instance
        :param model: LLM model instance
        :param temperature: Sampling temperature for LLM calls
        :param memory: Whether to include conversation history in the prompt
        :param num_history_messages: Number of recent messages to include when memory is enabled
        :param fallback_config: Optional config for fallback warning text
        """
        super().__init__(
            logger=logger,
            model=model,
            temperature=temperature,
            output_schema=None,
            memory=memory,
        )

        self.num_history_messages = num_history_messages
        self.fallback_config = fallback_config

    @override
    def _get_prompt_variables(self, state: BaseModel) -> dict:
        """Format prompt with the user's query."""
        return {"query": state.query}  # type: ignore

    @override
    def _update_state(self, result: Any, state: BaseModel) -> Dict[str, Any]:
        """Extract answer, append fallback warning if configured, update messages."""
        answer = ""

        # Add fallback warning if configured and query was domain-related
        fallback = self.fallback_config
        if fallback and fallback.enabled and fallback.warning:
            if getattr(state, "query_domain", "undefined") != "undefined":
                answer += f"\n\n{fallback.warning}\n\n"

        thought, answer_text = split_output_by_section(
            result.content, "<think>", "</think>"
        )
        answer += answer_text

        messages = list(state.messages)  # type: ignore
        result.content = answer
        messages.append(result)

        return {"messages": messages, "message_for_user": answer}

    # TODO: may need updating
    @override
    def _get_default_error_result(self) -> AIMessage:
        """Return default result when processing fails."""
        return AIMessage(content="")
