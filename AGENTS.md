# AGENTS.md - Neuroml_mcp Package

This file contains guidelines and commands for agentic coding agents working on the neuroml_mcp package.

## Project Overview

Neuroml_mcp is a Model Context Protocol (MCP) server that provides tools for NeuroML development. It uses FastMCP framework and implements a modular architecture with auto-discovered tools and sandboxed code execution.

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

# Run all pre-commit hooks manually
pre-commit run --all-files
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

### Server Operations
```bash
# Start the MCP server
nml-mcp

# Start with custom configuration (if implemented)
nml-mcp --config custom_config.yaml
```

## Code Style Guidelines

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

### Type Hints
- Use type hints consistently across function signatures
- Import from `typing` module for complex types
- Use `typing.Any` sparingly and prefer specific types when possible
- Return type annotations are mandatory for public APIs

### Error Handling
- Use specific exceptions rather than generic `Exception`
- Include descriptive error messages with context
- Use async context managers for resource cleanup
- Implement proper error propagation in sandbox operations

### Async/Await Patterns
- All sandbox operations should be async
- Use `async with` for context managers when available
- Implement proper async function signatures with `await`
- Handle async exceptions appropriately

## Architecture Guidelines

### Tool Development
- Tools must be functions ending with `_tool` suffix
- Use `@context.require()` decorator for dependencies when needed
- Include comprehensive docstrings with parameter types and examples
- Return structured data (dicts, Pydantic models) rather than raw strings
- Implement proper error handling for tool operations

### Sandbox Implementation
- Inherit from `AsyncSandbox` abstract base class
- Use `@singledispatchmethod` for handling different request types
- Implement proper security isolation for code execution
- Support both local and Docker-based execution environments
- Handle process timeouts and resource limits

### Server Configuration
- Use FastMCP framework conventions
- Implement health check endpoints
- Support both HTTP and Streamable HTTP transports
- Provide tool listing and discovery capabilities
- Handle graceful shutdowns and cleanup

## Configuration Management

### Package Configuration
- Primary configuration in `setup.cfg`
- Build configuration in `pyproject.toml`
- Use semantic versioning (currently 0.0.1)
- Support Python 3.12-3.13

### Environment Variables
- Use environment variables for configurable values
- Provide sensible defaults for all settings
- Document required environment variables in README
- Handle missing configuration gracefully

## Testing Guidelines

### Test Structure
- Use pytest with asyncio support
- Place tests in `tests/` directory
- Use descriptive test function names
- Mark tests requiring external services with `@pytest.mark.localonly`

### Test Patterns
- Use FastMCP client for integration testing
- Test tool discovery and registration
- Test sandbox isolation and security
- Mock external dependencies when possible
- Include both positive and negative test cases

## Documentation Requirements

### Code Documentation
- All public APIs must have docstrings
- Include parameter types, return types, and examples
- Use reStructuredText format for consistency
- Document async behavior and potential exceptions

### API Documentation
- Maintain tool documentation in docstrings
- Include usage examples for complex operations
- Document security considerations for sandbox operations
- Provide troubleshooting guidance

## Development Workflow

### Session Logging

At the start of each session, check the `.agents/` folder for previous session logs (named `YYYY-MM-DD.md`) to understand where work left off. Use the template in `.agents/README.md`.

### Pre-commit Requirements
- All code must pass ruff linting and formatting
- Import sorting is mandatory
- No trailing whitespace or large files
- Line endings must be Unix format

### Git Conventions
- Use conventional commit messages when possible
- Include relevant issue numbers in commit messages
- Keep commits focused and atomic
- Ensure all tests pass before pushing

## Security Considerations

### Sandbox Security
- Never execute untrusted code outside sandboxes
- Implement proper resource limits and timeouts
- Validate all inputs before execution
- Use read-only file systems when possible
- Monitor for suspicious activity patterns

### Code Safety
- Avoid eval() and exec() with user input
- Sanitize all file paths and inputs
- Implement proper access controls
- Use HTTPS for all external communications
- Never log sensitive information or credentials

## Common Patterns

### Tool Registration
```python
def my_tool_function(context: Context, param1: str, param2: int) -> Dict[str, Any]:
    """Tool description with comprehensive docstring.

    :param context: MCP context for the operation
    :param param1: Description of first parameter
    :param param2: Description of second parameter
    :returns: Dictionary with operation results
    """
    # Implementation here
    pass
```

### Sandbox Usage
```python
async with sandbox_manager.get_sandbox() as sandbox:
    result = await sandbox.execute_command(command, timeout=30)
    if result.return_code != 0:
        raise RuntimeError(f"Command failed: {result.stderr}")
```

This file should be updated as the project evolves. All agents should familiarize themselves with these guidelines before making changes to the codebase.
