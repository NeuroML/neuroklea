#!/usr/bin/env python3
"""
Tools router node

File: code_pkg/klea_code/nodes/tools_router.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import override

from neuroml_ai_utils.nodes.abstract import AbstractRouterNode

from klea_code.schemas import KleaCodeState


class ToolsRouter(AbstractRouterNode[KleaCodeState]):
    """Route based on tool call outputs."""

    def __init__(self, logger: logging.Logger):
        """Initialise the tools router node.

        :param logger: Logger instance
        """
        super().__init__(logger=logger)

    @override
    async def execute(self, state: KleaCodeState) -> str:
        """Route based on tool call outputs.

        :param state: The current state
        :return: The routing label
        """
        return ""
