#!/usr/bin/env python3
"""
Configurations for the API server

File: code_ai_pkg/neuroml_code_ai/api/conf.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    mcp_server_url: str = "http://127.0.0.1:8542"
    code_model: str = "ollama:qwen3.5:0.8b"
    reasoning_model: str = "ollama:qwen3.5:0.8b"
