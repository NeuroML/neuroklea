#!/usr/bin/env python3
"""
Configurations for the API server

File: code_pkg/klea_code/api/conf.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KLEA_CODE_")

    app_config_file: str = "klea_code.json"
    chat_model: str = "ollama:qwen3.5:0.8b"
    reasoning_model: str = "ollama:qwen3.5:0.8b"
    guard_model: str = "ollama:llama-guard3:1b"


class AppConfig(BaseModel):
    """Application configuration loaded from the JSON config file."""

    mcp_servers: dict[str, Any] = Field(default_factory=dict)
