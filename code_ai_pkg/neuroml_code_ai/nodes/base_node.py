#!/usr/bin/env python3
"""
Base node class

File: code_ai_pkg/neuroml_code_ai/base_node.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from pathlib import Path
from typing import Any, Generic, Type, TypeVar

from langchain_core.prompts import ChatPromptTemplate
from neuroml_ai_utils.llm import load_prompt
from neuroml_ai_utils.nodes import BaseLLMNode
from pydantic import BaseModel

from neuroml_code_ai import prompts

TSchema = TypeVar("TSchema", bound=BaseModel)

class BaseCodeAINode(BaseLLMNode[TSchema], Generic[TSchema]):
    """Base class for nodes that include memory"""

    def __init__(
        self,
        logger: logging.Logger,
        model: Any,
        temperature: float,
        output_schema: Type[TSchema],
        system_prompt_file: str,
        human_prompt_file: str,
        memory: bool = False,
    ):
        """Initialize with memory support"""
        super().__init__(logger, model, temperature, output_schema=output_schema)

        self.system_prompt_file = system_prompt_file
        self.human_prompt_file = human_prompt_file

        # TODO: used later when we implement memory
        self.memory = memory

    def _get_system_prompt(self) -> str:
        """Add other required bits to system prompt, like memory"""
        system_prompt = self._get_base_prompt(self.system_prompt_file)
        self.logger.debug(f"{system_prompt =}")
        return system_prompt

    def _get_human_prompt(self) -> str:
        """Add other required bits to system prompt, like memory"""
        human_prompt = self._get_base_prompt(self.human_prompt_file)

        if self.memory and hasattr(self, "_get_memory_addition"):
            memory_addition = self._get_memory_addition()
            return human_prompt + memory_addition
        self.logger.debug(f"{human_prompt =}")

        return human_prompt

    def _get_base_prompt(self, prompt_name: str) -> str:
        """Return the base system prompt"""
        loaded_prompt = load_prompt(
            prompt_name=prompt_name,
            prompt_registry_location=Path(prompts.__file__).parent,
        )
        return loaded_prompt

    def _create_prompt_template(
        self, system_prompt: str, human_prompt: str
    ) -> ChatPromptTemplate:
        """Create ChatPromptTemplate for this node"""
        prompt_template = ChatPromptTemplate(
            [("system", system_prompt), ("human", human_prompt)]
        )
        self.logger.debug(f"{prompt_template =}")
        return prompt_template

    def _get_memory_addition(self) -> str:
        """Override this to add memory-specific content"""
        # from .llm import add_memory_to_prompt
        # This will be overridden by subclasses to provide proper parameters
        return ""
