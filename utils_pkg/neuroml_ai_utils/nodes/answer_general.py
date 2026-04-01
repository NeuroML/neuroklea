#!/usr/bin/env python3
"""
Answer general question node

File: neuroml_ai_utils/nodes/answer_general.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from pathlib import Path
from typing import Any, Dict, override

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from ..llm import add_memory_to_prompt, load_prompt, split_output_by_section
from ..stores import FallbackConfig
from .base_nodes import BaseLLMNode


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
        prompt_registry_location: Path | None = None,
    ):
        """Initialise the general answer node.

        :param logger: Logger instance
        :param model: LLM model instance
        :param temperature: Sampling temperature for LLM calls
        :param memory: Whether to include conversation history in the prompt
        :param num_history_messages: Number of recent messages to include when memory is enabled
        :param fallback_config: Optional config for fallback warning text
        :param prompt_registry_location: Path to prompts directory (defaults to built-in)
        """
        super().__init__(logger, model, temperature, output_schema=None)

        self.memory = memory
        self.num_history_messages = num_history_messages
        self.fallback_config = fallback_config

        if prompt_registry_location is None:
            prompt_registry_location = Path(__file__).parent / "prompts"
        self.prompt_registry_location = prompt_registry_location

    @override
    def _get_system_prompt(self, state: BaseModel) -> str:
        """Load system prompt from file, optionally adding memory context."""
        system_prompt = load_prompt(
            prompt_name="answer_general_system",
            prompt_registry_location=self.prompt_registry_location,
        )

        if self.memory:
            system_prompt += add_memory_to_prompt(
                messages=state.messages,  # type: ignore
                context_summary=state.context_summary,  # type: ignore
                num_history_messages=self.num_history_messages,
            )

        return system_prompt

    @override
    def _get_human_prompt(self, state: BaseModel) -> str:
        """Load human prompt from file."""
        return load_prompt(
            prompt_name="answer_general_user",
            prompt_registry_location=self.prompt_registry_location,
        )

    @override
    def _create_prompt_template(
        self, system_prompt: str, human_prompt: str
    ) -> ChatPromptTemplate:
        """Create ChatPromptTemplate with system and human messages."""
        return ChatPromptTemplate([("system", system_prompt), ("human", human_prompt)])

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

    @override
    def _get_default_error_result(self) -> AIMessage:
        """Return default result when processing fails."""
        return AIMessage(content="")
