#!/usr/bin/env python3
"""
Main API script

File: code_pkg/klea_code/api/main.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from contextlib import asynccontextmanager

from cachetools import TTLCache
from fastapi import FastAPI

from klea_code.api.chat import chat_router
from klea_code.api.health import health_router
from klea_code.klea_code import KleaCode


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.is_ready = False
    app.state.sessions = TTLCache(maxsize=1000, ttl=7200)

    klea_code = KleaCode(memory=True)
    await klea_code.setup()

    app.state.klea_code = klea_code
    app.state.is_ready = True

    yield

    app.state.is_ready = False


app = FastAPI(lifespan=lifespan)
app.include_router(chat_router)
app.include_router(health_router)
