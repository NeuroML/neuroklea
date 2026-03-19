#!/usr/bin/env python3
"""
Chat end points

File: neuroml_ai/api/chat.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from fastapi import APIRouter, HTTPException, Request

chat_router = APIRouter()


@chat_router.post("/query")
async def query(request: Request, query: str):
    assistant = request.app.state.assistant
    try:
        result = await assistant.run_graph_invoke(query)
    except Exception as e:
        result = HTTPException(status_code=500, detail=str(e))

    return {"result": result}
