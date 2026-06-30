#!/usr/bin/env python3
"""
Server entry point for the Klea RAG API.

File: klea_rag/api/server.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import typer

serve_app = typer.Typer()


@serve_app.command()
def serve(
    host: str = "127.0.0.1",
    port: int = 8005,
    dev: bool = typer.Option(
        False, "--dev", help="Enable auto-reload (like fastapi dev)"
    ),
):
    """Run the Klea RAG API server."""
    # Lazy: uvicorn pulls in starlette/httptools/websockets etc.
    # Deferring to function body keeps --help fast.
    import uvicorn

    uvicorn.run(
        "klea_rag.api.main:app",
        host=host,
        port=port,
        reload=dev,
    )


if __name__ == "__main__":
    serve_app()
