#!/usr/bin/env python3
"""
Classify question domain node

File: rag_pkg/gen_rag/nodes/classify_question.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from pathlib import Path
from typing import Any, Dict, Type, override

from langchain_core.messages import AIMessage, HumanMessage
from neuroml_ai_utils.llm import load_prompt
from neuroml_ai_utils.nodes.base_nodes import BaseMemoryLLMNode
from neuroml_ai_utils.stores import VectorStores
from pydantic import BaseModel

from gen_rag.schemas import RAGState


class ClassifyQuestion(BaseMemoryLLMNode[RAGState]):
    """Classify a user query into a domain category.

    Uses an LLM to determine which domain the query belongs to, based on
    configured vector store domains. Appends conversation history to the
    system prompt when memory is enabled.
    """

    def __init__(
        self,
        logger: logging.Logger,
        model: Any,
        stores: VectorStores,
        query_domain_schema: Type[BaseModel],
        temperature: float = 0.3,
        memory: bool = False,
        prompt_registry_location: Path | None = None,
    ):
        """Initialise the classifier node.

        :param logger: Logger instance
        :param model: LLM model instance
        :param stores: VectorStores instance (provides domain configuration)
        :param query_domain_schema: Dynamically generated Pydantic schema for classification output
        :param temperature: Sampling temperature for LLM calls
        :param memory: Whether to include conversation history in the prompt
        :param prompt_registry_location: Path to prompts directory (defaults to built-in)
        """
        super().__init__(
            logger=logger,
            model=model,
            temperature=temperature,
            output_schema=query_domain_schema,
            memory=memory,
        )

        self.stores = stores

        if prompt_registry_location is None:
            prompt_registry_location = Path(__file__).parent / "prompts"
        self.prompt_registry_location = prompt_registry_location

    def _build_domain_str(self) -> str:
        """Build the domain classification string from vector store config."""
        domain_info = self.stores.vs_config.domains

        domain_str = self.stores.vs_config.pre_prompt
        domain_str += "\n\nCategories:\n\n"

        for d, info in domain_info.items():
            desc = info.description
            if not desc or len(desc) == 0:
                desc = f"if the question is about {d}"
            else:
                desc = f"if the question is about {desc}"
            domain_str += f"\n- {d}: {desc}"

        domain_str += "\n- undefined: otherwise"
        return domain_str

    @override
    def _get_system_prompt(self, state: RAGState) -> str:
        """Load base prompt, append domains, then rules, then optional memory."""
        system_prompt = load_prompt(
            prompt_name="classify_question_system",
            prompt_registry_location=self.prompt_registry_location,
        )
        system_prompt += f"\n\n## Domains\n{self._build_domain_str()}\n\n"

        if self.memory:
            system_prompt += self._get_memory_addition(state)

        self.logger.debug(f"{system_prompt =}")
        return system_prompt

    @override
    def _get_prompt_variables(self, state: RAGState) -> dict:
        """Format prompt with the user's query."""
        return {"query": state.query}

    @override
    def _update_state(self, result: Any, state: RAGState) -> Dict[str, Any]:
        """Extract classification result, append query to messages."""
        messages = list(state.messages)
        messages.append(HumanMessage(content=state.query))

        return {
            "query_domain": result.query_domain,
            "messages": messages,
        }

    # TODO: may need updating
    @override
    def _get_default_error_result(self) -> AIMessage:
        """Return default result when processing fails."""
        return AIMessage(content="")
