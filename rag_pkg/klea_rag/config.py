#!/usr/bin/env python3
"""
Config schema for the app

File: rag_pkg/klea_rag/config.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from typing import Any

from klea_utils.stores import VectorStoreInfo
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KLEA_RAG_")

    chat_model: str = "ollama:qwen2.5-coder:3b"
    guard_model: str = "ollama:llama-guard3:1b"
    embedding_model: str = "ollama:bge-m3:latest"
    app_config_file: str = "klea_rag.json"


class GeneralConfig(BaseModel):
    default_k: int = 5
    k_max: int = 10
    pre_prompt: str = ""
    non_domain_chat: bool = True
    fallback_to_training_data: bool = True
    fallback_warning: str = ""


class PerDomainConfig(BaseModel):
    """Configuration for a single domain."""

    description: str
    vector_stores: list[VectorStoreInfo]
    mcp_servers: dict[str, Any] = Field(default_factory=dict)


class AppConfig(BaseModel):
    general: GeneralConfig
    domains: dict[str, PerDomainConfig]
