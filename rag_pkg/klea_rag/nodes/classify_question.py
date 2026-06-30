#!/usr/bin/env python3
"""
Classify question domain node

File: rag_pkg/klea_rag/nodes/classify_question.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from textwrap import dedent
from typing import Any, Dict, Type, override

from klea_utils.nodes.base import BaseLLMNode
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from klea_rag.schemas import RAGState


# Type is calculated at runtime in orchestrator
class ClassifyQuestion[TSchema: BaseModel](BaseLLMNode[TSchema]):
    """Classify a user query into domain categories.

    Uses an LLM to determine which domains the query belongs to, based on
    configured domain metadata. Appends conversation history to the system
    prompt when memory is enabled.
    """

    def __init__(
        self,
        logger: logging.Logger,
        model: Any,
        domains: Dict[str, str],
        output_schema: Type[TSchema],
        temperature: float = 0.3,
        memory: bool = False,
        pre_prompt: str = "",
    ):
        """Initialise the classifier node.

        :param logger: Logger instance
        :param model: LLM model instance
        :param domains: Domain name to description mapping
        :param output_schema: Pydantic schema for classification output
        :param temperature: Sampling temperature for LLM calls
        :param memory: Whether to include conversation history in the prompt
        :param pre_prompt: Optional pre-prompt text for domain classification
        """
        super().__init__(
            logger=logger,
            model=model,
            temperature=temperature,
            output_schema=output_schema,
            memory=memory,
        )
        self.domains = domains
        self.pre_prompt = pre_prompt

    def _build_domain_str(self) -> str:
        """Build the domain classification string from domain metadata."""
        domain_str = self.pre_prompt
        domain_str += "\n\n## Domains\n\n"

        for d, desc in self.domains.items():
            if not desc or len(desc) == 0:
                desc = f"if the question is about {d}"
            else:
                desc = f"if the question is about {desc}"
            domain_str += f"\n- {d}: {desc}"

        domain_str += "\n- undefined: otherwise (if no other domain)"
        return domain_str

    @override
    def _get_system_prompt(self, state: RAGState) -> str:
        """Load base prompt, append domains, then rules, then optional memory."""
        system_prompt = self._load_prompt_file(f"{self.prompt_prefix}_system")

        # additional logic
        system_prompt += f"\n\n## Domains\n{self._build_domain_str()}\n\n"

        if self.memory:
            memory_addition = self._get_memory_addition(state)
            system_prompt += memory_addition

        if self.output_schema:
            system_prompt += dedent(
                f"""
                ## Output schema (strict)

                Respond in JSON following this schema:

                {str(self.output_schema_json).replace("{", "{{").replace("}", "}}")}
                """
            )

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

        domains = result.query_domains

        # limit domains to valid ones
        valid_domains = []
        for d in domains:
            if d in self.domains:
                valid_domains.append(d)

        # if no valid domains, default to "undefined"
        if len(valid_domains) == 0:
            valid_domains.append("undefined")

        # if there are multiple domains, but "undefined" is also included,
        # remove it: we assume that the other domains are valid domains
        if len(valid_domains) > 1 and "undefined" in valid_domains:
            valid_domains.remove("undefined")

        return {
            "query_domains": valid_domains,
            "messages": messages,
        }

    # TODO: may need updating
    @override
    def _get_default_error_result(self) -> AIMessage:
        """Return default result when processing fails."""
        return AIMessage(content="")
