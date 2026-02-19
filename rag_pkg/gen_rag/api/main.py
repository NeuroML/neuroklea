#!/usr/bin/env python3
"""
Main API script

File: rag_pkg/gen_rag/api/main.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from gen_rag.api.chat import chat_router
from gen_rag.api.health import health_router
from gen_rag.rag import RAG


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.is_ready = False

    rag_config_file = os.getenv("RAG_ENV_FILE", None)

    if rag_config_file:
        rag = RAG(config_file=rag_config_file, memory=True)
    else:
        rag = RAG(memory=True)

    await rag.setup()

    app.state.rag = rag
    app.state.is_ready = True

    yield

    app.state.is_ready = False


app = FastAPI(lifespan=lifespan)
app.include_router(chat_router)
app.include_router(health_router)
