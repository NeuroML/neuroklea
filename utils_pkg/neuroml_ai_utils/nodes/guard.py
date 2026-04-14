#!/usr/bin/env python3
"""
Guard node for safety checking

File: neuroml_ai_utils/nodes/guard.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, Dict, override

from langchain_core.messages import AIMessage
from pydantic import BaseModel

from .base import BaseLLMNode


class GuardNode(BaseLLMNode):
    """Safety guard node that checks if user queries are safe to process.

    Evaluates whether a query contains potentially harmful content
    and returns a routing decision ("safe" or "unsafe").

    Note: to be used with llama-guard, which always returns safe/unsafe.
    """

    def __init__(
        self,
        logger: logging.Logger,
        model: Any,
        temperature: float = 0.3,
        memory: bool = False,
    ):
        """Initialise the guard node.

        :param logger: Logger instance
        :param model: LLM model instance
        :param temperature: Sampling temperature for LLM calls
        :param memory: Whether to include conversation history in the prompt
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
        """Format prompt with the user's query."""
        return {"query": state.query}  # type: ignore

    @override
    def _update_state(self, result: AIMessage, state: BaseModel) -> Dict[str, Any]:
        """Check result for safety and return routing decision."""
        self.logger.debug(f"{result = }")

        if "unsafe" in result.content:
            return {"guard_decision": "unsafe"}
        return {"guard_decision": "safe"}

    @override
    def _get_default_error_result(self) -> str:
        """Unused: no schema in this node."""
        return "safe"
