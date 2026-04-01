#!/usr/bin/env python3
"""
Summarise conversation history node

File: neuroml_ai_utils/nodes/summarise_memory.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from pathlib import Path
from typing import Any, Dict, override

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from ..llm import get_last_n_conversations, load_prompt, split_output_by_section
from .base_nodes import BaseLLMNode


class SummariseMemoryNode(BaseLLMNode):
    """Node that summarises conversation history into a context summary.

    Uses _pre_exec() to skip execution if there aren't enough recent messages.
    Does NOT append the summary to messages — it's metadata, not a turn.
    """

    def __init__(
        self,
        logger: logging.Logger,
        model: Any,
        temperature: float = 0.3,
        num_recent_messages: int = 10,
        prompt_registry_location: Path | None = None,
    ):
        """Initialise the summarisation node.

        :param logger: Logger instance
        :param model: LLM model instance
        :param temperature: Sampling temperature for LLM calls
        :param num_recent_messages: Minimum number of messages before summarising
        :param prompt_registry_location: Path to prompts directory (defaults to built-in)
        """
        super().__init__(logger, model, temperature, output_schema=None)

        self.num_recent_messages = num_recent_messages
        self.conversation = ""

        if prompt_registry_location is None:
            prompt_registry_location = Path(__file__).parent / "prompts"
        self.prompt_registry_location = prompt_registry_location

    @override
    def _pre_exec(self, state: BaseModel) -> bool:
        """Skip if not enough recent conversations to summarise."""
        self.conversation, human_messages, ai_messages = get_last_n_conversations(
            state.messages,  # type: ignore
            state.summarised_till,  # type: ignore
            None,
        )
        conversations_num = len(human_messages) + len(ai_messages)

        if conversations_num < self.num_recent_messages:
            self.logger.debug(
                f"Not enough conversations to summarise yet: "
                f"{conversations_num}/{self.num_recent_messages}"
            )
            return False
        return True

    @override
    def _get_system_prompt(self, state: BaseModel) -> str:
        """Load system prompt from file."""
        return load_prompt(
            prompt_name="summarise_memory",
            prompt_registry_location=self.prompt_registry_location,
        )

    @override
    def _get_human_prompt(self, state: BaseModel) -> str:
        """Load human prompt from file."""
        return load_prompt(
            prompt_name="summarise_memory_human",
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
        """Format prompt with conversation data."""
        return {
            "old_summary": state.context_summary,  # type: ignore
            "conversation": self.conversation,
        }

    @override
    def _update_state(self, result: Any, state: BaseModel) -> Dict[str, Any]:
        """Extract summary from raw AIMessage output."""
        self.logger.debug(f"Current history summary is:\n{result.content}")
        thought, answer = split_output_by_section(result.content, "<think>", "</think>")
        return {
            "context_summary": answer,
            "summarised_till": len(state.messages),  # type: ignore
        }

    @override
    def _get_default_error_result(self) -> AIMessage:
        """Return default result when processing fails."""
        return AIMessage(content="")
