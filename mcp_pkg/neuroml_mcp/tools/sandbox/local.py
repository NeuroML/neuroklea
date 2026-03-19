#!/usr/bin/env python3
"""
A sans-sandbox executor. Only to be used for development.
Be VERY careful when using this one.

File: neuroml_mcp/tools/sandbox/nope.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import asyncio
from functools import singledispatchmethod
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

from neuroml_mcp.tools.sandbox.sandbox import (
    AsyncSandbox,
    CmdResult,
    PatchCommand,
    RunCommand,
    RunPythonCode,
)


class LocalSandbox(AsyncSandbox):
    """A local execution sandbox"""

    def __init__(self, path: str):
        """Init

        :param path: path where all commands for this context manager will be run

        """
        self.wdir = path

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    @singledispatchmethod
    async def run(self, request):
        """Dummy fallback"""
        raise NotImplementedError("Not implemented")

    @run.register  # type: ignore
    async def _(self, request: RunPythonCode) -> CmdResult:
        with NamedTemporaryFile(prefix="nml_ai", mode="w") as f:
            print(request.code, file=f)

            process = await asyncio.create_subprocess_exec(
                "python",
                f.name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()
            assert process.returncode is not None
            return CmdResult(
                stderr=stderr.decode(),
                stdout=stdout.decode(),
                returncode=process.returncode,
                data={},
            )

    @run.register  # type: ignore
    async def _(self, request: RunCommand) -> CmdResult:
        process = await asyncio.create_subprocess_shell(
            " ".join(request.command),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()
        assert process.returncode is not None
        return CmdResult(
            stderr=stderr.decode(),
            stdout=stdout.decode(),
            returncode=process.returncode,
            data={},
        )

    @run.register  # type: ignore
    async def _(self, request: PatchCommand) -> CmdResult:
        with TemporaryDirectory() as tmpdir:
            cwd = Path(tmpdir)
            code_file = cwd / "code.py"
            patch_file = cwd / "patch.diff"

            code_file.write_text(request.base)
            patch_file.write_text(request.patch)

            cmd = RunCommand(["patch", str(code_file), "-i", str(patch_file)])
            result = await self.run(cmd)

            success = result.returncode == 0
            updated_code = code_file.read_text() if success else request.base
            result.data = {"code": updated_code}

            return result
