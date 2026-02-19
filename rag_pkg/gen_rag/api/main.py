#!/usr/bin/env python3
"""
Main API script

File: rag_pkg/gen_rag/api/main.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from gen_rag.api.chat import chat_router
from gen_rag.api.health import health_router
from gen_rag.rag import RAG


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.is_ready = False

    rag = RAG(config_file="rag.env", memory=True)
    await rag.setup()

    app.state.rag = rag
    app.state.is_ready = True

    yield

    app.state.is_ready = False


app = FastAPI(lifespan=lifespan)
app.include_router(chat_router)
app.include_router(health_router)
