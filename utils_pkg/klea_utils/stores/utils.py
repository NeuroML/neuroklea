#!/usr/bin/env python3
"""
Vector store utilities

File: klea_utils/stores/utils.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

from langchain_core.documents import Document

_INTERNAL_META_KEYS = {"file_name", "source_path", "file_hash", "headings"}


def serialize_vs_retrieval(
    reference_material: Dict[str, List[Tuple[Document, float]]],
) -> str:
    """Serialize vector store retrieval results into text for use in prompt context.

    Documents are sorted by relevance score within each group.
    Uses Docling ``HybridChunker`` metadata format:

    - ``headings``: list of heading hierarchy (most specific last)
    - ``file_name``: source filename
    - ``source_path``: full path to source file
    - Optional custom keys from the ``--metadata-map`` (e.g., ``url``)

    :param reference_material: Dict mapping query/domain to list of (doc, score) tuples
    :returns: Formatted string representation of references
    """
    serialized = ""
    for q, sorted_refs in reference_material.items():
        ctr = 1
        serialized += f"## {q}\n"
        for r, score in sorted_refs:
            headings = r.metadata.get("headings", [])
            file_name = r.metadata.get("file_name", "")
            heading_str = " > ".join(headings) if headings else "(no heading)"
            if file_name:
                heading_str = f"[{file_name}] {heading_str}"

            score_str = f" (relevance score: {score:.4f})"
            serialized += (
                f"\n### Document {ctr}/{len(sorted_refs)}: {heading_str}{score_str}\n"
            )
            custom_meta = {
                k: v for k, v in r.metadata.items() if k not in _INTERNAL_META_KEYS
            }
            if custom_meta:
                meta_str = " | ".join(f"{k}={v}" for k, v in custom_meta.items())
                serialized += f"Metadata: {meta_str}\n"
            serialized += r.page_content
            ctr += 1

    return serialized


def instantiate_vector_store(
    path: str,
    name: str,
    embeddings,
    logger: logging.Logger,
    create: bool = False,
):
    """Instantiate a vector store based on the URI scheme in path.

    Expected format: ``"scheme:location"``.

    If ``create`` is ``True``, the store is created if it does not exist
    (relevant for ChromaDB which requires a local directory).  For
    Qdrant and PGVector the flag is a no-op --- collections are created
    on first write.

    :param path: URI-style string with scheme prefix
        (e.g. ``"chroma:/path/to/dir"``, ``"qdrant:http://localhost:6333"``,
        ``"pgvector:postgresql://localhost/db"``)
    :param name: Collection name for the vector store
    :param embeddings: Embedding function to use
    :param logger: Logger instance
    :param create: If ``True``, allow creating a new store
    :returns: Instantiated LangChain VectorStore
    :raises ValueError: If the scheme is missing or unknown
    :raises FileNotFoundError: If ``create`` is ``False`` and a local
        ChromaDB store does not exist
    """
    scheme, sep, location = path.partition(":")
    if not sep:
        raise ValueError(
            f"Invalid vector store path '{path}': "
            f"expected format 'scheme:location' (e.g. 'chroma:/path/to/store')"
        )

    match scheme.lower():
        case "chroma":
            try:
                import chromadb
                from langchain_chroma import Chroma
            except ImportError:
                raise ImportError(
                    "ChromaDB backend not installed. "
                    "Install: pip install klea_utils[chroma]"
                ) from None

            store_dir = Path(location)
            if not store_dir.is_absolute():
                store_dir = Path.cwd() / store_dir
                logger.debug(f"Store path made absolute relative to cwd: {store_dir}")

            if create:
                store_dir.mkdir(parents=True, exist_ok=True)
            else:
                if not store_dir.is_dir():
                    logger.error(f"Could not find folder: {store_dir}")
                    raise FileNotFoundError(f"Could not find folder: {store_dir}")

                store_db = store_dir / "chroma.sqlite3"
                if not store_db.is_file():
                    raise FileNotFoundError(f"ChromaDB not found at {store_db}")

            logger.debug(
                f"Loading Chroma vector store '{name}' from {store_dir.absolute()}"
            )

            settings = chromadb.config.Settings(
                is_persistent=True,
                persist_directory=str(store_dir.absolute()),
                anonymized_telemetry=False,
            )
            return Chroma(
                collection_name=name,
                embedding_function=embeddings,
                client_settings=settings,
            )

        case "qdrant":
            try:
                from langchain_qdrant import QdrantVectorStore
                from qdrant_client import QdrantClient
            except ImportError:
                raise ImportError(
                    "Qdrant backend not installed. "
                    "Install: pip install klea_utils[qdrant]"
                ) from None

            client = QdrantClient(url=location)
            logger.debug(f"Loading Qdrant vector store '{name}' at {location}")
            return QdrantVectorStore(
                client=client,
                collection_name=name,
                embedding=embeddings,
            )

        case "pgvector":
            try:
                from langchain_postgres import PGVector
            except ImportError:
                raise ImportError(
                    "PGVector backend not installed. "
                    "Install: pip install klea_utils[pgvector]"
                ) from None

            logger.debug(
                f"Loading PGVector vector store '{name}' with connection {location}"
            )
            return PGVector(
                collection_name=name,
                embeddings=embeddings,
                connection=location,
            )

        case _:
            raise ValueError(
                f"Unknown vector store scheme '{scheme}'. "
                f"Supported: chroma, qdrant, pgvector"
            )
