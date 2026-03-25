# AGENTS.md - Code AI Package

AI assisted coding for NeuroML using LangChain/LangGraph.

## Package Overview

Package: `neuroml_code_ai`
CLI entry: `nml-code`

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
pytest tests/test_code_ai.py

# Run tests with verbose output
pytest -v
```

## Architecture

### Package Structure
```
neuroml_code_ai/
├── api/         # API endpoints
├── nodes/       # LangGraph nodes
├── prompts/     # Prompt templates
├── schemas.py   # Pydantic schemas
├── ui/          # CLI interface
└── code_ai.py  # Main orchestration
```

### Key Technologies
- LangChain/LangGraph for agent orchestration
- FastMCP for MCP integration
- Typer for CLI
- Streamlit for UI

## Code Style

### File Organization
- **Header**: All Python files should have a copyright header
- **Docstrings**: Use reStructuredText format
- **Module structure**: `__init__.py` files should be minimal or empty

### Import Conventions
```python
# 1. Standard library imports
import asyncio
import os
from typing import Any, Dict, List

# 2. Third-party imports
from langchain_core.messages import HumanMessage, AIMessage
from fastmcp import Context
from pydantic import BaseModel

# 3. Local imports
from neuroml_code_ai.schemas import CodeGenRequest
from neuroml_code_ai.nodes import generate_code_node
```

### Naming Conventions
- **Functions**: snake_case (`generate_code_node`, `validate_neuroml_model`)
- **Classes**: PascalCase (`CodeGenRequest`, `LangGraphAgent`)
- **Variables**: snake_case (`model_name`, `code_output`)
- **Constants**: UPPER_CASE (`DEFAULT_MODEL`, `MAX_RETRIES`)
