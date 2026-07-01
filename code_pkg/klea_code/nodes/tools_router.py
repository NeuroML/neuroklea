#!/usr/bin/env python3
"""
Tools router node

File: code_pkg/klea_code/nodes/tools_router.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from typing import override

from klea_utils.nodes.abstract import AbstractRouterNode

from klea_code.schemas import KleaCodeState


class ToolsRouter(AbstractRouterNode[KleaCodeState]):
    """Route based on tool call outputs."""

    def __init__(self, logger: logging.Logger, label: str):
        """Initialise the tools router node.

        :param logger: Logger instance
        :param label: Human-readable label for UI progress display
        """
        super().__init__(logger, label)

    @override
    async def execute(self, state: KleaCodeState) -> str:
        """Route based on tool call outputs.

        :param state: The current state
        :return: The routing label
        """
        self.write_custom_stream({"type": "progress", "node": self.label})
        return ""
