# AGENTS.md - RAG Package

Generic RAG (Retrieval Augmented Generation) implementation for NeuroML.

## Package Overview

Package: `gen_rag`
CLI entry: `nml-gen-rag`

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
pytest tests/test_rag.py

# Run tests with verbose output
pytest -v
```

## Architecture

### Package Structure
```
gen_rag/
├── api/         # API endpoints
├── config.py    # Configuration
├── data/        # Data loading and processing
├── nodes/       # LangGraph nodes
├── prompts/     # Prompt templates
├── rag.py       # Main RAG logic
├── schemas.py   # Pydantic schemas
├── stores.py    # Vector stores
└── ui/          # CLI interface
```

### Key Technologies
- LangChain for RAG implementation
- Chroma for vector storage
- LangGraph for orchestration
- HuggingFace embeddings

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
from langchain_core.documents import Document
from langchain_chroma import Chroma
from pydantic import BaseModel

# 3. Local imports
from gen_rag.schemas import RAGRequest
from gen_rag.nodes import retrieve_node
```

### Naming Conventions
- **Functions**: snake_case (`retrieve_documents`, `create_vector_store`)
- **Classes**: PascalCase (`RAGRequest`, `DocumentStore`)
- **Variables**: snake_case (`query`, `documents`, `embedding_model`)
- **Constants**: UPPER_CASE (`DEFAULT_COLLECTION`, `MAX_RESULTS`)

### Vector Store Configuration
- Default collection name should be configurable via environment
- Support both local Chroma and HuggingFace deployments
- Implement proper error handling for connection failures
