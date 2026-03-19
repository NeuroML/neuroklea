#!/usr/bin/env python3
"""
Configuration for AI assistant

File: neuroml_ai/neuroml_ai/config.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    guard_model: str = "ollama:llama-guard3:1b"
    chat_model: str = "ollama:qwen2.5-coder:3b"
