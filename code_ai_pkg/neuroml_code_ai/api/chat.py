#!/usr/bin/env python3
"""
Chat end points

File: code_ai_pkg/neuroml_code_ai/api/chat.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from fastapi import APIRouter, Request

chat_router = APIRouter()


@chat_router.post("/query")
async def query(request: Request, query: str):
    code_ai = request.app.state.code_ai
    return {"result": await code_ai.run_graph_invoke(query)}
