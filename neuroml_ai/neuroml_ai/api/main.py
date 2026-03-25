#!/usr/bin/env python3
"""
Main API script

File: neuroml_ai/api/main.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import os
from contextlib import asynccontextmanager

from cachetools import TTLCache
from fastapi import FastAPI
from fastmcp import Client

from neuroml_ai.api.chat import chat_router
from neuroml_ai.api.health import health_router
from neuroml_ai.assistant import NML_Assistant


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.is_ready = False
    app.state.sessions = TTLCache(maxsize=1000, ttl=7200)

    assistant = NML_Assistant()
    await assistant.setup()

    app.state.assistant = assistant
    app.state.is_ready = True

    yield

    app.state.is_ready = False


# Currently NO-OP
async def lifespan1(app: FastAPI):
    app.state.is_ready = False

    client_url = "http://127.0.0.1:8542/mcp"
    mcp_client = Client(client_url)

    chat_model = os.environ.get("GEN_RAG_CHAT_MODEL", "ollama:qwen3:1.7b")
    vs_config_file = os.environ.get(
        "GEN_RAG_VS_CONFIG",
        "/home/asinha/Documents/02_Code/00_mine/NeuroML/software/neuroml-ai/rag_pkg/vector-stores.json",
    )

    # check that client is up
    async with mcp_client:
        await mcp_client.ping()
        tools = await mcp_client.list_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")

    assistant = NML_Assistant(vs_config_file=vs_config_file, chat_model=chat_model)
    await assistant.setup()

    app.state.assistant = assistant
    app.state.mcp = Client
    app.state.is_ready = True

    yield

    app.state.is_ready = False


app = FastAPI(lifespan=lifespan)
app.include_router(chat_router)
app.include_router(health_router)
