#!/usr/bin/env python3
"""
Retrieve information node

File: rag_pkg/klea_rag/nodes/retrieve_info.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, Dict, override

from klea_utils.nodes.abstract import AbstractLangGraphNode
from klea_utils.stores.retrieval import VSRetriever

from klea_rag.schemas import RAGState


class RetrieveInfoNode(AbstractLangGraphNode[RAGState, Dict[str, Any]]):
    """Retrieve reference material from vector stores.

    Queries the vector stores for all domains in the query_domains list using
    the same retrieval query, ranks results by relevance score, and keeps the
    top N references for each domain. Optionally increments k when asked to
    retrieve more info.
    """

    def __init__(
        self,
        logger: logging.Logger,
        stores: VSRetriever | None,
        num_refs_max: int = 10,
    ):
        """Initialise the retrieval node.

        :param logger: Logger instance
        :param stores: VSRetriever instance for retrieval (None skips retrieval)
        :param num_refs_max: Maximum number of references to keep per domain
        """
        super().__init__(logger)
        self.stores = stores
        self.num_refs_max = num_refs_max

    @override
    async def execute(self, state: RAGState) -> Dict[str, Any]:
        """Retrieve and rank reference material."""
        if self.stores is None:
            self.logger.debug("No vector stores configured, skipping retrieval")
            return {}

        reference_material = state.reference_material
        cleaned_query = state.retrieval_query

        self.logger.debug(f"retrieval query: {cleaned_query}")

        # Check if evaluator requested more info
        if state.text_response_eval.next_step == "retrieve_more_info":
            self.stores.inc_k()

        # Retrieve from vector stores for all domains
        for domain_name in state.query_domains:
            # Skip undefined domain
            if domain_name == "undefined":
                continue

            res = self.stores.retrieve(domain_name=domain_name, query=cleaned_query)

            # Rank by relevance score, keep top N
            sorted_res = sorted(res, key=lambda tup: tup[1], reverse=True)
            new_ref = {domain_name: sorted_res[: self.num_refs_max]}

            reference_material.update(new_ref)

        self.logger.debug(f"{reference_material =}")

        return {"reference_material": reference_material}
