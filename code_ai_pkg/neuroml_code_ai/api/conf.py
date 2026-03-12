#!/usr/bin/env python3
"""
Configurations for the API server

File: code_ai_pkg/neuroml_code_ai/api/conf.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    nml_mcp_server_url: str = "http://127.0.0.1:8542"
    nml_ai_code_model: str = "ollama:qwen3.5:35b"
    nml_ai_reasoning_model: str = "ollama:qwen3.5:35b"


nml_code_ai_settings = Settings()
