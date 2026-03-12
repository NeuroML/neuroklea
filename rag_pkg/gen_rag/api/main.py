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

    chat_model = os.environ.get(
        "GEN_RAG_CHAT_MODEL",
        "vllm:Qwen/Qwen3-14B-GGUF:Q4_K_M",
    )
    vs_config_file = os.environ.get(
        "GEN_RAG_VS_CONFIG",
        "/home/hengyezhu/neuroml-ai/rag_pkg/vector-stores.json",
    )

    rag = RAG(chat_model=chat_model, vs_config_file=vs_config_file, memory=True)
    await rag.setup()

    app.state.rag = rag
    app.state.is_ready = True

    yield

    app.state.is_ready = False


app = FastAPI(lifespan=lifespan)
app.include_router(chat_router)
app.include_router(health_router)
