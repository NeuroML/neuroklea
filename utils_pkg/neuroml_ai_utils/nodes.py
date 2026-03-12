#!/usr/bin/env python3
"""
Base node class for LangGraph processing nodes

File: neuroml_ai_utils/nodes.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, Type, TypeVar, Union

from langchain_core.prompt_values import PromptValue
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from .llm import parse_output_with_thought

TSchema = TypeVar("TSchema", bound=BaseModel)

class BaseLLMNode(ABC, Generic[TSchema]):
    """Abstract base class for LangGraph nodes that use LLMs"""

    def __init__(
        self,
        logger: logging.Logger,
        model_inst: Any,
        temperature: float,
        output_schema: Optional[Type[TSchema]] = None,
    ):
        """Initialize with logger and model"""
        self.logger = logger
        self.model_inst = model_inst
        self.temperature = temperature
        self.output_schema = output_schema

    async def execute(self, state: BaseModel) -> Dict[str, Any]:
        """Template method defining standard execution flow"""
        self.logger.debug(f"{state =}")

        human_prompt = self._get_human_prompt()
        system_prompt = self._get_system_prompt()
        template = self._create_prompt_template(system_prompt, human_prompt)
        variables = self._get_prompt_variables(state)
        prompt = self._invoke_prompt(template, variables)
        llm = self._configure_llm()
        output = self._invoke_llm(llm, prompt)

        result = self._process_output(output)
        state_updates = self._update_state(result, state)

        self.logger.debug(f"{state_updates =}")

        return state_updates

    # but can be overridden
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
    def _get_human_prompt(self) -> str:
        """Return human prompt for this node"""
        pass

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Return system prompt for this node"""
        pass

    @abstractmethod
    def _create_prompt_template(
        self, system_prompt: str, human_prompt: str
    ) -> ChatPromptTemplate:
        """Create ChatPromptTemplate for this node"""
        pass

    @abstractmethod
    def _get_prompt_variables(self, state: BaseModel) -> dict:
        """Format prompt with state-specific parameters"""
        pass

    @abstractmethod
    def _update_state(self, result: TSchema, state: BaseModel) -> Dict[str, Any]:
        """Update and return state dictionary"""
        pass

    @abstractmethod
    def _get_default_error_result(self) -> TSchema:
        """Return default result when processing fails"""
        pass
