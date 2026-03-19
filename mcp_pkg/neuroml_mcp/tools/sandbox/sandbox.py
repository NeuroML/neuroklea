#!/usr/bin/env python3
"""
Sandbox interface to be used by implementations

File: mcp/tools/sandbox/sandbox.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from abc import ABC, abstractmethod
from contextlib import AbstractAsyncContextManager
from typing import Any, List

from pydantic.dataclasses import dataclass


@dataclass
class RunPythonCode:
    """Run Python code"""

    code: str


@dataclass
class RunCommand:
    """Run a command provided as a list of strings"""

    command: List[str]


@dataclass
class PatchCommand:
    """Patch the base file with the given patch"""

    base: str
    patch: str


@dataclass
class CmdResult:
    """Result from a command execution"""

    stdout: str
    stderr: str
    returncode: int
    data: dict[Any, Any]


class AsyncSandbox(AbstractAsyncContextManager, ABC):
    """Abstract async context manager class"""

    @abstractmethod
    async def run(self, request):
        """Runner method to be implemented"""
        raise NotImplementedError("Not implemented")
