#!/usr/bin/env python3
"""
Base node classes for LangGraph processing nodes

File: neuroml_ai_utils/nodes/base_nodes.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from .llm import add_memory_to_prompt, load_prompt, parse_output_with_thought


class BaseLangGraphNode[TSchema: BaseModel, TReturn](ABC):
    """Abstract base class for all LangGraph nodes.

    Generic over TReturn to support both state-updating nodes (Dict[str, Any])
    and router nodes (str).

    Provides a consistent interface: all nodes have a logger and an
    execute(state) method.
    """

    def __init__(self, logger: logging.Logger):
        """Initialise with a logger.

        :param logger: Logger instance
        """
        self.logger = logger

    @abstractmethod
    async def execute(self, state: TSchema) -> TReturn:
        """Execute this node and return the result.

        :param state: Current graph state
        :returns: State updates (dict) or routing label (str)
        """
        ...


class BaseLLMNode[TSchema: BaseModel](BaseLangGraphNode[TSchema, Dict[str, Any]]):
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
        output_schema: Optional[Type[TSchema]] = None,
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
        self.output_schema = output_schema

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

    def _pre_exec(self, state: BaseModel) -> bool:
        """Pre-execution check. Override to conditionally skip node execution.

        Return False to skip execution (returns empty dict).
        Return True (default) to proceed with the standard flow.
        """
        return True

    def _get_output_schema(self) -> Optional[Type[TSchema]]:
        """Return Pydantic schema for structured output if required"""
        return self.output_schema

    def _configure_llm(self) -> Runnable:
        """Configure LLM with structured output"""
        output_schema = self._get_output_schema()
        if output_schema:
            return self.model_inst.with_structured_output(
                output_schema, method="json_schema", include_raw=True
            )
        else:
            return self.model_inst

    def _invoke_llm(self, llm: Runnable, prompt: PromptValue) -> Any:
        """Invoke LLM with default temperature - can be overridden"""
        output = llm.invoke(
            prompt, config={"configurable": {"temperature": self.temperature}}
        )
        self.logger.debug(f"{output = }")
        return output

    def _process_output(self, output: Any) -> Any:
        """Common output processing with error handling"""
        output_schema = self._get_output_schema()
        if output_schema:
            if output["parsing_error"]:
                self.logger.warning(
                    f"LLM parsing error, using fallback: {output['parsing_error']}"
                )
                result = parse_output_with_thought(output["raw"], output_schema)
            else:
                result = output["parsed"]
                if isinstance(result, dict):
                    result = output_schema(**result)
                else:
                    if not isinstance(result, output_schema):
                        self.logger.critical(f"Unexpected output type: {type(result)}")
                        result = self._get_default_error_result()

            self.logger.debug(f"Processed output: {result}")
        else:
            result = output
            self.logger.debug(f"No output schema. Output: {result}")

        return result

    def _invoke_prompt(
        self, prompt_template: ChatPromptTemplate, variables: Union[Any, Dict[str, Any]]
    ) -> PromptValue:
        """Format prompt with state-specific parameters"""
        prompt = prompt_template.invoke(variables)
        self.logger.debug(f"{prompt =}")
        return prompt

    # Abstract methods to be implemented by subclasses
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


class BaseMemoryLLMNode[TSchema: BaseModel](BaseLLMNode[TSchema]):
    """Base class for LangGraph nodes that load prompts from files.

    Extends BaseLLMNode with:
    - File-based prompt loading via load_prompt()
    - Optional memory support (appends memory content to human prompt)
    - Configurable prompt registry location (package-specific prompt directories)
    """

    def __init__(
        self,
        logger: logging.Logger,
        model: Any,
        temperature: float,
        output_schema: Type[TSchema],
        system_prompt_file: str,
        human_prompt_file: str,
        prompt_registry_location: Path,
        memory: bool = False,
    ):
        """Initialize with file-based prompt loading and memory support.

        :param logger: Logger instance
        :param model: LLM model instance
        :param temperature: Sampling temperature for LLM calls
        :param output_schema: Pydantic schema for structured output
        :param system_prompt_file: Name of the system prompt file (no extension)
        :param human_prompt_file: Name of the human prompt file (no extension)
        :param prompt_registry_location: Path to the prompts directory
        :param memory: Whether to append memory content to the human prompt
        """
        super().__init__(logger, model, temperature, output_schema=output_schema)

        self.system_prompt_file = system_prompt_file
        self.human_prompt_file = human_prompt_file
        self.prompt_registry_location = prompt_registry_location
        self.memory = memory
        self.num_recent_messages = 10

    def _get_system_prompt(self, state: BaseModel) -> str:
        """Load system prompt from file."""
        system_prompt = self._load_prompt_file(self.system_prompt_file)
        self.logger.debug(f"{system_prompt =}")
        return system_prompt

    def _get_human_prompt(self, state: BaseModel) -> str:
        """Load human prompt from file, optionally appending memory content."""
        human_prompt = self._load_prompt_file(self.human_prompt_file)

        if self.memory:
            memory_addition = self._get_memory_addition(state)
            return human_prompt + memory_addition

        self.logger.debug(f"{human_prompt =}")
        return human_prompt

    def _load_prompt_file(self, prompt_name: str) -> str:
        """Load a prompt file from the registry.

        :param prompt_name: Prompt file name (without extension)
        :returns: Prompt text content
        """
        return load_prompt(
            prompt_name=prompt_name,
            prompt_registry_location=self.prompt_registry_location,
        )

    def _create_prompt_template(
        self, system_prompt: str, human_prompt: str
    ) -> ChatPromptTemplate:
        """Create ChatPromptTemplate with system and human messages."""
        prompt_template = ChatPromptTemplate(
            [("system", system_prompt), ("human", human_prompt)]
        )
        self.logger.debug(f"{prompt_template =}")
        return prompt_template

    def _get_memory_addition(self, state: BaseModel) -> str:
        """Hook for subclasses to inject memory content into the human prompt.

        Override this method to provide memory-specific content.
        The default implementation returns an empty string.
        """
        return add_memory_to_prompt(
            messages=state.messages,  # type: ignore
            context_summary=state.context_summary,  # type: ignore
            num_recent_messages=self.num_recent_messages,
        )


class BaseRouterNode[TSchema: BaseModel](BaseLangGraphNode[TSchema, str]):
    """Base class for LangGraph router nodes.

    Router nodes inspect the state and return a string label that determines
    which edge to follow next. Used with ``add_conditional_edges()``.
    """

    @abstractmethod
    async def execute(self, state: TSchema) -> str:
        """Return the routing label (edge name) based on state."""
        ...
