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

logging.basicConfig()
logging.root.setLevel(logging.WARNING)

app = typer.Typer(help="Create vector stores from documents")


@app.command()
def build(
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
    metadata_map_path: str = typer.Option(
        None,
        "--metadata-map",
        "-M",
        help="JSON file mapping heading text to dicts of metadata "
        "key-value pairs (with optional DEFAULT fallback)",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-process all files even if unchanged"
    ),
):
    """Full pipeline: chunk, embed, and write to a vector store.

    Processes all files in SOURCE_DIR: converts them with Docling,
    chunks them, embeds them, and writes to the vector store.
    Processed chunks are cached in ``<source_dir>/.klea-cache/`` so
    subsequent runs (e.g. with ``--metadata-map``) skip conversion.

    The optional ``--metadata-map`` / ``-M`` flag accepts a JSON file
    where each key is a heading string and each value is a dict of
    metadata key-value pairs.  The most specific heading match wins; a
    ``DEFAULT`` key provides fallback.

    Example metadata-map.json:

        {
            "C. elegans tissue morphology": {
                "url": "https://example.com/worm",
                "category": "paper"
            },
            "DEFAULT": {
                "url": "https://example.com"
            }
        }
    """
    logger = setup_logger("klea-vs-create")

    logger.info(
        f"Building vector store '{collection_name}' at {store_path}"
        f"\n  Source: {source_dir}"
        f"\n  Model: {embedding_model}"
        f"\n  Max tokens: {max_tokens}"
        f"\n  Metadata map: {metadata_map_path or '(none)'}"
    )

    try:
        # Lazy: importing VSBuilder pulls in ingestion.py -> llm.py ->
        # langchain_huggingface/langchain_ollama, stores/utils.py ->
        # chromadb/qdrant etc.  Deferring to function body keeps
        # --help fast (Python only needs the function signature).
        from klea_utils.stores.ingestion import VSBuilder

        builder = VSBuilder(
            embedding_model=embedding_model,
            logger=logger,
            max_tokens=max_tokens,
        )
        builder.build(
            source_dir=source_dir,
            store_uri=store_path,
            collection_name=collection_name,
            force=force,
            metadata_map_path=metadata_map_path,
        )
        logger.info(f"Done -- collection '{collection_name}' is ready")
    except Exception as e:
        logger.error(f"Failed: {e}")
        raise typer.Exit(1) from None


@app.command()
def chunk(
    source_dir: str = typer.Argument(help="Directory containing source documents"),
    max_tokens: int = typer.Option(
        450, "--max-tokens", help="Maximum tokens per chunk"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-process all files even if unchanged"
    ),
):
    """Chunk and cache documents without writing to a vector store.

    Converts all files in SOURCE_DIR with Docling, chunks them, and
    caches the result in ``<source_dir>/.klea-cache/``.  Also writes a
    ``metadata-map.template.json`` file with every unique heading chain
    found.  Fill in the ``{}`` placeholders and pass the file to
    ``klea-vs-create store --metadata-map``.
    """
    logger = setup_logger("klea-vs-create")

    logger.info(f"Chunking documents in {source_dir}\n  Max tokens: {max_tokens}")

    try:
        # Lazy: importing VSBuilder pulls in ingestion.py -> llm.py ->
        # langchain_huggingface/langchain_ollama, stores/utils.py ->
        # chromadb/qdrant etc.  Deferring to function body keeps
        # --help fast (Python only needs the function signature).
        from pathlib import Path

        from klea_utils.stores.ingestion import VSBuilder

        builder = VSBuilder(
            embedding_model="",  # not needed for chunking only
            logger=logger,
            max_tokens=max_tokens,
        )
        source_path = Path(source_dir).resolve()
        if not source_path.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source_path}")

        _, heading_chains = builder.chunk_all(source_path, force=force)
        builder.write_heading_template(heading_chains, source_path)
        logger.info("Chunking complete -- cache is ready")
    except Exception as e:
        logger.error(f"Failed: {e}")
        raise typer.Exit(1) from None


@app.command()
def store(
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
        450, "--max-tokens", help="Maximum tokens per chunk (for files not yet cached)"
    ),
    metadata_map_path: str = typer.Option(
        None,
        "--metadata-map",
        "-M",
        help="JSON file mapping heading text to dicts of metadata "
        "key-value pairs (with optional DEFAULT fallback)",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-process all files even if unchanged"
    ),
):
    """Write cached document chunks to a vector store.

    Reads previously cached chunks from ``<source_dir>/.klea-cache/``,
    optionally applies a metadata map, and writes them to the vector
    store.  Unseen files are converted and chunked on the fly.

    Run ``klea-vs-create chunk`` first to populate the cache and
    generate a ``metadata-map.template.json``.
    """
    logger = setup_logger("klea-vs-create")

    logger.info(
        f"Storing cached chunks to '{collection_name}' at {store_path}"
        f"\n  Source: {source_dir}"
        f"\n  Model: {embedding_model}"
        f"\n  Metadata map: {metadata_map_path or '(none)'}"
    )

    try:
        from pathlib import Path

        from klea_utils.stores.ingestion import VSBuilder

        builder = VSBuilder(
            embedding_model=embedding_model,
            logger=logger,
            max_tokens=max_tokens,
        )
        source_path = Path(source_dir).resolve()
        if not source_path.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source_path}")

        metadata_map = None
        if metadata_map_path:
            metadata_map = builder._load_metadata_map(metadata_map_path)

        results, _ = builder.chunk_all(
            source_path, metadata_map=metadata_map, force=force
        )
        builder.store_all(results, store_path, collection_name, force=force)
        logger.info(f"Done -- collection '{collection_name}' is ready")
    except Exception as e:
        logger.error(f"Failed: {e}")
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()
