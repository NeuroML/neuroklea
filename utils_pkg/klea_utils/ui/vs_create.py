#!/usr/bin/env python3
"""
CLI for creating vector stores from documents

File: klea_utils/ui/vs_create.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

import typer

from klea_utils.plogging import setup_logger
from klea_utils.stores.ingestion import VSBuilder

logging.basicConfig()
logging.root.setLevel(logging.WARNING)

app = typer.Typer(help="Create vector stores from documents")


@app.command()
def create(
    source_dir: str = typer.Argument(help="Directory containing source documents"),
    collection_name: str = typer.Option(
        ..., "--collection", "-n", help="Collection name for the vector store"
    ),
    store_path: str = typer.Option(
        ..., "--store", "-s", help="Vector store URI (e.g. chroma:/path/to/store)"
    ),
    embedding_model: str = typer.Option(
        "ollama:bge-m3:latest",
        "--model",
        "-m",
        help="Embedding model identifier",
    ),
    max_tokens: int = typer.Option(
        450, "--max-tokens", help="Maximum tokens per chunk"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-process all files even if unchanged"
    ),
):
    """Convert documents in SOURCE_DIR, chunk, embed, and write to a vector store."""
    logger = setup_logger("klea-vs-create")

    logger.info(
        f"Building vector store '{collection_name}' at {store_path}"
        f"\n  Source: {source_dir}"
        f"\n  Model: {embedding_model}"
        f"\n  Max tokens: {max_tokens}"
    )

    try:
        builder = VSBuilder(
            embedding_model=embedding_model,
            logger=logger,
            max_tokens=max_tokens,
        )
        builder.setup()
        builder.build(
            source_dir=source_dir,
            store_uri=store_path,
            collection_name=collection_name,
            force=force,
        )
        logger.info(f"Done — collection '{collection_name}' is ready")
    except Exception as e:
        logger.error(f"Failed: {e}")
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()
