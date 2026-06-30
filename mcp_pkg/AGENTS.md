# AGENTS.md - MCP Package

Model Context Protocol (MCP) server providing tools for NeuroML development.

## Package Overview

Package: `neuroml_mcp`
CLI entry: `nml-mcp`

## Development Commands

### Building and Installation
```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install -e .[dev]
```

### Linting and Formatting
```bash
# Run ruff for linting and fixing
ruff check . --fix
ruff format .

# Sort imports specifically
ruff check . --select I --fix
```

### Testing
```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_mcp.py

# Run a specific test function
pytest tests/test_mcp.py::test_tool_discovery

# Run tests with verbose output
pytest -v

# Run tests marked as local only (require LLM access)
pytest -m localonly

# Run tests excluding local only
pytest -m "not localonly"
```

## Architecture

### Package Structure
```
neuroml_mcp/
├── server/      # FastMCP server
├── tools/       # Auto-discovered tools
├── sandbox/     # Sandboxed code execution
├── utils.py     # Utilities
└── __init__.py
```

### Key Technologies
- FastMCP framework
- PyNeuroML for NeuroML handling
- Starlette for HTTP
- Sandbox isolation for code execution

### Tool Development
- Tools must be functions ending with `_tool` suffix
- Use `@context.require()` decorator for dependencies
- Include comprehensive docstrings with parameter types and examples
- Return structured data (dicts, Pydantic models) rather than raw strings

### Sandbox Implementation
- Inherit from `AsyncSandbox` abstract base class
- Use `@singledispatchmethod` for handling different request types
- Implement proper security isolation for code execution
- Support both local and Docker-based execution environments
- Handle process timeouts and resource limits

## Code Style

### File Organization
- **Header**: All Python files must start with `#!/usr/bin/env python3` shebang
- **Copyright**: Follow with copyright format: `# Copyright 2025 Ankur Sinha <ankursinha@fedoraproject.org>`
- **Docstrings**: Use reStructuredText format with parameter and return type documentation
- **Module structure**: `__init__.py` files should be minimal or empty

### Import Conventions
```python
# 1. Standard library imports
import asyncio
import os
from typing import Any, Dict, List

# 2. Third-party imports
from fastmcp import ClientContext, Context
from pydantic import BaseModel
from starlette.applications import Starlette

# 3. Local imports
from neuroml_mcp.sandbox.local import LocalSandbox
from neuroml_mcp.utils import register_all_tools
```

### Naming Conventions
- **Functions**: snake_case with descriptive names (`create_new_NeuroML_model_tool`)
- **Classes**: PascalCase (`LocalSandbox`, `RunCommand`)
- **Variables**: snake_case (`tool_context`, `sandbox_manager`)
- **Constants**: UPPER_CASE (`DEFAULT_TIMEOUT`, `MAX_RETRIES`)
- **Tool functions**: Must end with `_tool` suffix for auto-discovery
- **Private functions**: Prefix with underscore (`_internal_helper`)

### Async/Await Patterns
- All sandbox operations should be async
- Use `async with` for context managers when available
- Implement proper async function signatures with `await`
- Handle async exceptions appropriately

## Security Considerations

### Sandbox Security
- Never execute untrusted code outside sandboxes
- Implement proper resource limits and timeouts
- Validate all inputs before execution
- Use read-only file systems when possible
- Monitor for suspicious activity patterns

### Sandbox Usage Pattern
```python
async with sandbox_manager.get_sandbox() as sandbox:
    result = await sandbox.execute_command(command, timeout=30)
    if result.return_code != 0:
        raise RuntimeError(f"Command failed: {result.stderr}")
```
