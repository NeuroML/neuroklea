#!/usr/bin/env python3
"""
Route evaluator node

File: rag_pkg/klea_rag/nodes/route_evaluator.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

from neuroml_ai_utils.nodes.abstract import AbstractRouterNode
from neuroml_ai_utils.stores import VectorStores

from klea_rag.schemas import RAGState


class RouteEvaluator(AbstractRouterNode):
    """Route based on Evaluator node results"""

    def __init__(self, logger: logging.Logger, stores: VectorStores):
        """Initialise the evaluator node.

        :param logger: Logger instance
        :param stores: Vector Stores
        """
        super().__init__(
            logger=logger,
        )
        self.stores = stores

    def execute(self, state: RAGState):
        """Route based on state, set by evaluator node."""
        self.logger.debug(f"{state =}")
        resp = state.text_response_eval
        next_step = resp.next_step

        if next_step == "continue" and (
            resp.coverage >= 0.5
            and resp.confidence >= 0.5
            and resp.relevance >= 0.5
            and resp.groundedness >= 0.5
            and resp.coherence >= 0.5
            and resp.conciseness >= 0.5
        ):
            self.stores.reset_k()
            self.logger.debug("returning: continue")
            return "continue"
        elif not state.query_modified and (
            next_step == "modify_query" or resp.coverage < 0.3
        ):
            self.logger.debug("returning: modify_query")
            return "modify_query"
        elif next_step == "retrieve_more_info" or (
            resp.coverage >= 0.5 and resp.confidence < 0.5
        ):
            # limit what max k we can have, otherwise, we end up pulling the
            # whole store..
            if self.stores.inc_k():
                self.logger.debug("returning: retrieve_more_info")
                return "retrieve_more_info"
            else:
                # we are already at max context, so we need to modify the query
                # to get a better result if possible
                if not state.query_modified:
                    self.logger.debug("returning: modify_query")
                    return "modify_query"
                # if we've already modified query, fallback to training data if
                # possible, otherwise ask for clarification
                else:
                    if self.stores.vs_config.fallback_to_training_data:
                        self.logger.debug("returning: fallback")
                        return "fallback"
                    else:
                        self.logger.debug("returning: undefined")
                        return "undefined"
        elif next_step == "rewrite_answer" or (
            resp.coverage >= 0.5
            and resp.confidence >= 0.5
            and (
                resp.relevance < 0.5
                and resp.groundedness < 0.5
                and resp.coherence < 0.5
                and resp.conciseness < 0.5
            )
        ):
            self.logger.debug("returning: rewrite_answer")
            return "rewrite_answer"
        # all other cases: fallback to training data if enabled, otherwise ask for clarification
        else:
            if self.stores.vs_config.fallback_to_training_data:
                self.logger.debug("returning: fallback")
                return "fallback"
            else:
                self.logger.debug("returning: undefined")
                return "undefined"
