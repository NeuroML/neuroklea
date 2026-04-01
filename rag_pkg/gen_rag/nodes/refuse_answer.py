#!/usr/bin/env python3
"""
Refuse to answer node

File: rag_pkg/gen_rag/nodes/refuse_answer.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, Dict, override

from neuroml_ai_utils.nodes.base_nodes import BaseLangGraphNode
from neuroml_ai_utils.stores import VectorStores

from gen_rag.schemas import RAGState


class RefuseAnswer(BaseLangGraphNode[RAGState, Dict[str, Any]]):
    """Refuse to answer queries outside permitted domains."""

    def __init__(self, logger: logging.Logger, stores: VectorStores):
        """Initialise with logger and vector stores reference.

        :param logger: Logger instance
        :param stores: VectorStores instance (used to get permitted domains)
        """
        super().__init__(logger)
        self.stores = stores

    @override
    async def execute(self, state: RAGState) -> Dict[str, Any]:
        """Return refusal message listing permitted domains."""
        msg = "Sorry. I cannot answer this query as it does not fall into my permitted domains. Available domains are:\n"
        msg += "\n- ".join([""] + self.stores.domains)
        msg += "\n\n\nPlease try another query."

        return {"message_for_user": msg}
