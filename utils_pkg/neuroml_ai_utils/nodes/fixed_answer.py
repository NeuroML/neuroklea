#!/usr/bin/env python3
"""
Provide a fixed answer.

File: rag_pkg/gen_rag/nodes/fixed_answer.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any, Dict, override

from gen_rag.schemas import RAGState
from pydantic import BaseModel

from neuroml_ai_utils.nodes.abstract import AbstractLangGraphNode


class FixedAnswer(AbstractLangGraphNode[BaseModel, Dict[str, Any]]):
    """Provide a fixed answer"""

    def __init__(self, logger: logging.Logger, state_attr: str, message: str):
        """Initialise with logger and message to return.

        :param logger: Logger instance
        :param message: str message to return
        """
        super().__init__(logger)
        self.message = message
        self.state_attr = state_attr

    @override
    async def execute(self, state: RAGState) -> Dict[str, Any]:
        """Return fixed message."""
        self.logger.debug({self.state_attr: self.message})
        return {self.state_attr: self.message}
