#!/usr/bin/env python3
"""
Chat end points

File: neuroml_ai/api/chat.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

chat_router = APIRouter()


class ChatPayload(BaseModel):
    query: str
    session_id: str


@chat_router.post("/query")
async def query(request: Request, payload: ChatPayload):
    thread_id = payload.session_id
    sessions = request.app.state.sessions
    assistant = request.app.state.assistant

    if thread_id not in sessions:
        sessions[thread_id] = True

    try:
        result = await assistant.run_graph_invoke(payload.query, thread_id)
    except Exception as e:
        result = HTTPException(status_code=500, detail=str(e))

    return {"result": result}
