#!/usr/bin/env python3
"""
Summarise conversation history node

File: klea_utils/nodes/summarise_memory.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, Dict, override

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from ..llm import get_last_n_conversations, split_output_by_section
from .base import BaseLLMNode


class SummariseMemoryNode(BaseLLMNode):
    """Node that summarises conversation history into a context summary.

    Uses _pre_exec() to skip execution if there aren't enough recent messages.
    Does NOT append the summary to messages -- it's metadata, not a turn.

    Expects state to have the following fields:

    - messages: list of messages
    - summarised_till: index of messages that have been summarised already
    - context_summary: previous memory/context summary

    """

    def __init__(
        self,
        logger: logging.Logger,
        label: str,
        model: Any,
        temperature: float = 0.3,
        summarisation_threshold: int = 10,
        memory: bool = False,
    ):
        """Initialise the summarisation node.

        :param logger: Logger instance
        :param label: Human-readable label for UI progress display
        :param model: LLM model instance
        :param temperature: Sampling temperature for LLM calls
        :param summarisation_threshold: Minimum number of messages before summarising
        :param memory: Whether to include conversation history in the prompt
        """
        super().__init__(
            logger=logger,
            label=label,
            model=model,
            temperature=temperature,
            output_schema=None,
            memory=memory,
        )

        self.summarisation_threshold = summarisation_threshold
        self.conversation = ""

    @override
    def _pre_exec(self, state: BaseModel) -> bool:
        """Skip if not enough recent conversations to summarise."""
        self.conversation, human_messages, ai_messages = get_last_n_conversations(
            state.messages,  # type: ignore
            state.summarised_till,  # type: ignore
            None,
        )
        conversations_num = len(human_messages) + len(ai_messages)

        if conversations_num < self.summarisation_threshold:
            self.logger.debug(
                f"Not enough conversations to summarise yet: "
                f"{conversations_num}/{self.summarisation_threshold}"
            )
            return False
        return True

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

    # TODO: may need updating
    @override
    def _get_default_error_result(self) -> AIMessage:
        """Return default result when processing fails."""
        return AIMessage(content="")
