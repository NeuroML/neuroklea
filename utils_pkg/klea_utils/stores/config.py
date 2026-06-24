#!/usr/bin/env python3
"""
Vector store configuration models

File: klea_utils/stores/config.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from typing import Any, Dict

from pydantic import BaseModel


class VectorStoreInfo(BaseModel):
    """Information about a single vector store."""

    name: str
    path: str
    loaded_object: Any | None = None


class PerDomainConfig(BaseModel):
    """Configuration for a single domain."""

    description: str
    vector_stores: list[VectorStoreInfo]


class VectorStoresConfig(BaseModel):
    """Top-level vector stores configuration."""

    default_k: int = 5
    k_max: int = 10
    embedding_model: str
    domains: Dict[str, PerDomainConfig]
