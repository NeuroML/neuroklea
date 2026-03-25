# AGENTS.md - Neuroml-ai Package

Main AI assistant application for NeuroML using LangChain/LangGraph.

## Package Overview

Package: `neuroml-ai`
CLI entry: `nml-ai`

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
pytest tests/test_assistant.py

# Run tests with verbose output
pytest -v
```

## Architecture

### Package Structure
```
neuroml_ai/
├── api/         # API endpoints
├── ui/          # CLI interface
├── assistant.py # Main assistant logic
├── config.py    # Configuration
└── schemas.py   # Pydantic schemas
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
from neuroml_ai.schemas import AssistantRequest
from neuroml_ai.assistant import AssistantNode
```

### Naming Conventions
- **Functions**: snake_case (`process_user_message`, `validate_neuroml`)
- **Classes**: PascalCase (`AssistantRequest`, `NeuroMLAssistant`)
- **Variables**: snake_case (`user_input`, `response_text`)
- **Constants**: UPPER_CASE (`DEFAULT_MODEL`, `MAX_TOKENS`)
