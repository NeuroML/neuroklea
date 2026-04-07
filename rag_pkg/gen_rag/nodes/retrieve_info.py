#!/usr/bin/env python3
"""
Retrieve information node

File: rag_pkg/gen_rag/nodes/retrieve_info.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, Dict, override

from neuroml_ai_utils.nodes.abstract import AbstractLangGraphNode
from neuroml_ai_utils.stores import VectorStores

from gen_rag.schemas import RAGState


class RetrieveInfoNode(AbstractLangGraphNode[RAGState, Dict[str, Any]]):
    """Retrieve reference material from vector stores.

    Queries the vector stores for a given domain and query, ranks results by
    relevance score, and keeps the top N references. Optionally increments k
    when asked to retrieve more info.
    """

    def __init__(
        self,
        logger: logging.Logger,
        stores: VectorStores,
        num_refs_max: int = 10,
    ):
        """Initialise the retrieval node.

        :param logger: Logger instance
        :param stores: VectorStores instance for retrieval
        :param num_refs_max: Maximum number of references to keep per domain
        """
        super().__init__(logger)
        self.stores = stores
        self.num_refs_max = num_refs_max

    @override
    async def execute(self, state: RAGState) -> Dict[str, Any]:
        """Retrieve and rank reference material."""
        reference_material = state.reference_material
        cleaned_query = state.retrieval_query

        self.logger.debug(f"retrieval query: {cleaned_query}")

        # Check if evaluator requested more info
        if state.text_response_eval.next_step == "retrieve_more_info":
            self.stores.inc_k()

        # Retrieve from vector stores
        res = self.stores.retrieve(domain_name=state.query_domain, query=cleaned_query)

        # Rank by relevance score, keep top N
        sorted_res = sorted(res, key=lambda tup: tup[1], reverse=True)
        new_ref = {state.query_domain: sorted_res[: self.num_refs_max]}

        reference_material.update(new_ref)
        self.logger.debug(f"{reference_material =}")

        return {"reference_material": reference_material}
