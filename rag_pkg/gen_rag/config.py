#!/usr/bin/env python3
"""
Config schema for the app

File: rag_pkg/gen_rag/config.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    chat_model: str = "ollama:qwen2.5-coder:3b"
    non_domain_chat: bool = True
    vs_config_file: str = "vector_stores.json"
