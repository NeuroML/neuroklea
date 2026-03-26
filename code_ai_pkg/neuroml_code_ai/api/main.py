#!/usr/bin/env python3
"""
Main API script

File: code_ai_pkg/neuroml_code_ai/api/main.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from contextlib import asynccontextmanager

from cachetools import TTLCache
from fastapi import FastAPI

from neuroml_code_ai.api.chat import chat_router
from neuroml_code_ai.api.health import health_router
from neuroml_code_ai.code_ai import CodeAI


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.is_ready = False
    app.state.sessions = TTLCache(maxsize=1000, ttl=7200)

    code_ai = CodeAI(memory=True)
    await code_ai.setup()

    app.state.code_ai = code_ai
    app.state.is_ready = True

    yield

    app.state.is_ready = False


app = FastAPI(lifespan=lifespan)
app.include_router(chat_router)
app.include_router(health_router)
