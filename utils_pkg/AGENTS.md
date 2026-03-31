# AGENTS.md - Utils Package

Shared utilities for NeuroML AI packages.

## Package Overview

Package: `neuroml_ai_utils`

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
pytest tests/test_utils.py

# Run tests with verbose output
pytest -v
```

## Architecture

### Package Structure
```
neuroml_ai_utils/
├── api.py       # Shared API utilities
├── errors.py    # Custom exception classes
├── graph.py     # BaseLangGraph abstract class
├── llm.py       # LLM utilities
├── logging.py   # Logging configuration
└── nodes.py     # Shared LangGraph nodes
```

### Key Technologies
- LangChain-core for base utilities
- HuggingFace for embeddings
- httpx for HTTP requests
- tenacity for retry logic

## Code Style

### File Organization
- **Header**: All Python files should have a copyright header
- **Docstrings**: Use reStructuredText format
- **Module structure**: `__init__.py` files should be minimal or empty

### Import Conventions
```python
# 1. Standard library imports
import logging
from typing import Any, Dict, List

# 2. Third-party imports
from langchain_core.messages import BaseMessage
from langchain_huggingface import HuggingFaceEmbeddings
import httpx

# 3. Local imports
from neuroml_ai_utils.logging import setup_logging
from neuroml_ai_utils.llm import get_default_model
```

### Naming Conventions
- **Functions**: snake_case (`setup_logging`, `get_default_model`)
- **Classes**: PascalCase (`NeuroMLLogger`, `LLMConfig`)
- **Variables**: snake_case (`model_name`, `api_key`)
- **Constants**: UPPER_CASE (`DEFAULT_MODEL`, `LOG_FORMAT`)

### Guidelines
- Utilities should be framework-agnostic where possible
- Provide sensible defaults but allow configuration
- Document environment variable dependencies
- Include type hints for all public functions
