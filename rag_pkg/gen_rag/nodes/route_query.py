#!/usr/bin/env python3
"""
Route query node

File: rag_pkg/gen_rag/nodes/route_query.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

from neuroml_ai_utils.nodes.abstract import AbstractRouterNode
from neuroml_ai_utils.stores import VectorStores

from gen_rag.schemas import RAGState


class RouteQuery(AbstractRouterNode):
    """Route based on Query node results"""

    def __init__(
        self,
        logger: logging.Logger,
        stores: VectorStores,
        non_domain_chat: bool = False,
    ):
        """Initialise the node.

        :param logger: Logger instance
        :param stores: Vector Stores
        :param non_domain_chat: boolean if non domain chat is enabled
        """
        super().__init__(
            logger=logger,
        )
        self.stores = stores
        self.non_domain_chat = non_domain_chat

    def execute(self, state: RAGState):
        """Route based on query domain, set by query classifier node."""
        self.logger.debug(f"{state =}")
        query_domain = state.query_domain

        if query_domain in self.stores.domains and query_domain != "undefined":
            res = "domain_query"
        else:
            if self.non_domain_chat:
                res = "non_domain_query"
            else:
                res = "non_domain_refuse"
        self.logger.debug(f"{res = }")
        return res
