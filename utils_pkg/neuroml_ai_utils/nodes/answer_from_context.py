#!/usr/bin/env python3
"""
Generate an answer from provided reference material

File: utils_pkg/neuroml_ai_utils/nodes/answer_from_context.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import Any

from .base_nodes import BaseMemoryLLMNode


# TODO: complete
class AnswerFromContext(BaseMemoryLLMNode):
    """Generate an answer from the provided context"""

    def __init__(
        self,
        logger: logging.Logger,
        model: Any,
        temperature: float = 0.3,
        memory: bool = False,
    ):
        """TODO: to be defined.

        :param logger: TODO
        :param memory: TODO
        :param : TODO

        """
        super().__init__(
            logger=logger,
            model=model,
            temperature=temperature,
            output_schema=None,
            memory=memory,
        )
