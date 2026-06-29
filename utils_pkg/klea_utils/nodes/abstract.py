#!/usr/bin/env python3
"""
Abstract node classes for LangGraph processing nodes

File: klea_utils/nodes/abstract.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Type, final

from langchain.messages import AIMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel


class AbstractLangGraphNode[TSchema: BaseModel, TReturn](ABC):
    """Abstract base class for all LangGraph nodes.

    Generic over TReturn to support both state-updating nodes (Dict[str, Any])
    and other nodes, e.g., router nodes (str) and tool caller nodes.

    Provides a consistent interface: all nodes have a logger and an
    execute(state) method.
    """

    def __init__(self, logger: logging.Logger):
        """Initialise

        Creates a new hierarchical logger.

        :param logger: Parent logger instance (used to derive child logger name)
        """
        # Child logger -- inherits the parent's dual-stream handlers
        # (set up by BaseLangGraph via plogging.setup_logger) through
        # propagation, so this class does NOT configure its own
        # handlers.
        self.logger = logging.getLogger(f"{logger.name}.{self.__class__.__name__}")

    @abstractmethod
    async def execute(self, state: TSchema) -> TReturn:
        """Execute this node and return the result.

        :param state: Current graph state
        :returns: State updates (dict) or routing label (str)
        """
        ...


class AbstractLLMNode[TSchema: BaseModel](
    AbstractLangGraphNode[TSchema, Dict[str, Any]]
):
    """Abstract base class for LangGraph nodes that use LLMs.

    Implements a template execution flow:
    1. Pre-execution check (optional skip)
    2. Build prompt (system + human)
    3. Invoke LLM
    4. Process output (structured or raw)
    5. Update state
    """

    def __init__(
        self,
        logger: logging.Logger,
        model_inst: Any,
        temperature: float,
        output_schema: Type[TSchema] | None = None,
    ):
        """Initialize with logger and model.

        :param logger: Logger instance
        :param model_inst: LLM model instance
        :param temperature: Sampling temperature
        :param output_schema: Pydantic schema for structured output
        """
        super().__init__(logger)
        self.model_inst = model_inst
        self.temperature = temperature
        self._output_schema = output_schema

    @final
    async def execute(self, state: BaseModel) -> Dict[str, Any]:
        """Template method defining standard execution flow"""
        self.logger.debug(f"{state =}")

        if not self._pre_exec(state):
            self.logger.debug("Pre-exec check failed, skipping execution")
            return {}

        human_prompt = self._get_human_prompt(state)
        system_prompt = self._get_system_prompt(state)
        template = self._create_prompt_template(system_prompt, human_prompt)
        variables = self._get_prompt_variables(state)
        prompt = self._invoke_prompt(template, variables)
        llm = self._configure_llm()
        output = self._invoke_llm(llm, prompt)

        result = self._process_output(output)
        state_updates = self._update_state(result, state)

        self.logger.debug(f"{state_updates =}")

        return state_updates

    @abstractmethod
    def _pre_exec(self, state: BaseModel) -> bool:
        """Pre-execution check. Override to conditionally skip node execution.

        Return False to skip execution (returns empty dict).
        Return True (default) to proceed with the standard flow.
        """
        ...

    @abstractmethod
    def _configure_llm(self) -> Runnable:
        """Configure LLM with structured output"""
        ...

    @abstractmethod
    def _invoke_llm(
        self, llm: Runnable, prompt: PromptValue
    ) -> AIMessage | dict[str, Any]:
        """Invoke LLM with default temperature - can be overridden"""
        ...

    @abstractmethod
    def _process_output(self, output: AIMessage | dict[str, Any]) -> Any:
        """Common output processing with error handling"""
        ...

    @abstractmethod
    def _invoke_prompt(
        self, prompt_template: ChatPromptTemplate, variables: Any | Dict[str, Any]
    ) -> PromptValue:
        """Format prompt with state-specific parameters"""
        ...

    @abstractmethod
    def _get_human_prompt(self, state: BaseModel) -> str:
        """Return human prompt for this node"""
        ...

    @abstractmethod
    def _get_system_prompt(self, state: BaseModel) -> str:
        """Return system prompt for this node"""
        ...

    @abstractmethod
    def _create_prompt_template(
        self, system_prompt: str, human_prompt: str
    ) -> ChatPromptTemplate:
        """Create ChatPromptTemplate for this node"""
        ...

    @abstractmethod
    def _get_prompt_variables(self, state: BaseModel) -> dict:
        """Format prompt with state-specific parameters"""
        ...

    @abstractmethod
    def _update_state(self, result: Any, state: BaseModel) -> Dict[str, Any]:
        """Update and return state dictionary"""
        ...

    @abstractmethod
    def _get_default_error_result(self) -> Any:
        """Return default result when processing fails"""
        ...


class AbstractRouterNode[TSchema: BaseModel](AbstractLangGraphNode[TSchema, str]):
    """Abstract class for LangGraph router nodes.

    Router nodes inspect the state and return a string label that determines
    which edge to follow next. Used with ``add_conditional_edges()``.
    """

    @abstractmethod
    async def execute(self, state: TSchema) -> str:
        """Return the routing label (edge name) based on state."""
        ...
