#!/usr/bin/env python3
"""
Config schema for the app

File: rag_pkg/gen_rag/config.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    guard_model: str = "ollama:llama-guard3:1b"
    chat_model: str = "ollama:qwen2.5-coder:3b"
    vs_config_file: str = "vector_stores.json"
    answer_non_domain: str = "yes"
