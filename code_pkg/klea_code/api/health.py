#!/usr/bin/env python3
"""
Health check end points

File: code_pkg/klea_code/api/health.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from fastapi import APIRouter, Request, Response, status

health_router = APIRouter()


@health_router.get("/health/live")
async def liveness():
    return {"status": "alive"}


@health_router.get("/health/ready")
async def readiness(request: Request):
    is_ready = getattr(request.app.state, "is_ready", False)

    if is_ready:
        return {"status": "ready"}

    return Response(
        content="Service not ready", status_code=status.HTTP_503_SERVICE_UNAVAILABLE
    )
