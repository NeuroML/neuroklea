#!/usr/bin/env python3
"""
Chat end points

File: rag_pkg/gen_rag/api/chat.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import traceback

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

logging.basicConfig(
    format="%(name)s (%(levelname)s) >>> %(message)s\n", level=logging.WARNING
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

chat_router = APIRouter()


class ChatPayload(BaseModel):
    query: str
    session_id: str


@chat_router.post("/query")
async def query(request: Request, payload: ChatPayload):
    thread_id = payload.session_id
    sessions = request.app.state.sessions
    rag = request.app.state.rag

    if thread_id not in sessions:
        sessions[thread_id] = True

    try:
        result = await rag.run_graph_invoke(payload.query, thread_id)
    except Exception as e:
        detail = f"{e}\n{traceback.format_exc()}"
        result = HTTPException(status_code=500, detail=detail)

        logger.error(detail)

    return {"result": result}
