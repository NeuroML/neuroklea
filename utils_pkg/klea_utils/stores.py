#!/usr/bin/env python3
"""
Vector stores management for RAG systems

File: klea_utils/stores.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.documents import Document
from pydantic import BaseModel

from .llm import setup_embedding


def serialize_vs_retrieval(
    reference_material: Dict[str, List[Tuple[Document, float]]],
) -> str:
    """Serialize vector store retrieval results into text for use in prompt context.

    Documents are sorted by relevance score within each group.

    :param reference_material: Dict mapping query/domain to list of (doc, score) tuples
    :returns: Formatted string representation of references
    """
    serialized = ""
    for q, sorted_refs in reference_material.items():
        ctr = 1
        serialized += f"## {q}\n"
        for r, score in sorted_refs:
            metadata = [
                f"{key}: {val}"
                for key, val in r.metadata.items()
                if "header" in key.lower()
            ]
            metadata_str = f"### Document {ctr}/{len(sorted_refs)}: " + " | ".join(
                metadata
            )
            serialized += "\n" + f"{metadata_str}\n"
            url = r.metadata.get("url", None)
            if url:
                serialized += f"Reference URL: {url}\n"
            serialized += r.page_content
            ctr += 1

    return serialized


class VectorStoreInfo(BaseModel):
    """Information about a single vector store."""

    name: str
    path: str
    loaded_object: Any | None = None


class PerDomainConfig(BaseModel):
    """Configuration for a single domain."""

    description: str
    vector_stores: list[VectorStoreInfo]


class VectorStoresConfig(BaseModel):
    """Top-level vector stores configuration."""

    default_k: int = 5
    k_max: int = 10
    embedding_model: str
    domains: Dict[str, PerDomainConfig]


class VectorStores:
    """Manages domain-specific vector stores.

    Loads vector stores on demand per domain and provides similarity search
    retrieval across multiple stores within a domain.

    Store paths use a URI-style scheme prefix to identify the backend:

    - ``chroma:/path/to/dir``  ---  ChromaDB (persistent, local disk)
    - ``qdrant:http://host:port``  ---  Qdrant (remote HTTP)
    - ``pgvector:postgresql://host/db``  ---  PGVector (PostgreSQL)
    """

    def __init__(self, vs_config: VectorStoresConfig, logger: logging.Logger):
        """Initialise vector stores manager.

        :param logger: Logger instance (injected from orchestrator)
        """
        self.default_k = 5
        self.k_max = 10
        self.k = self.default_k
        self.sim_thresh = 0.15
        self.embeddings = None
        self.vs_config: VectorStoresConfig = vs_config
        self.logger = logging.getLogger(f"{logger.name}.{self.__class__.__name__}")
        self.embedding_model: str | None = None

    def setup(self) -> None:
        """Load configuration and initialise embedding model."""
        self._load_config()
        self.embeddings = setup_embedding(self.embedding_model, self.logger)

        # Extract model name for collection naming
        assert self.embedding_model
        if self.embedding_model.lower().startswith("huggingface:"):
            self.embedding_model = (
                self.embedding_model.replace("huggingface:", "")
                .replace(":cheapest", "")
                .replace(":fastest", "")
            )
            splits = self.embedding_model.split("/")
            self.embedding_model = "".join(splits[1:])
        elif self.embedding_model.lower().startswith("ollama:"):
            self.embedding_model = self.embedding_model.replace("ollama:", "")

    def _load_config(self) -> None:
        """Load domains from the configuration file."""
        self.embedding_model = self.vs_config.embedding_model
        self.default_k = self.vs_config.default_k
        self.k_max = self.vs_config.k_max
        self.logger.debug(f"{self.vs_config =}")

        assert self.embedding_model

    def inc_k(self, inc: int = 1) -> bool:
        """Increase k by inc.

        :param inc: Amount to increase k by
        :returns: True if k was increased, False if already at max
        """
        if (self.k + inc) <= self.k_max:
            self.k += inc
            self.logger.debug(f"k increased to {self.k =}")
            return True
        return False

    def reset_k(self) -> None:
        """Reset k to default value."""
        self.k = self.default_k
        self.logger.debug(f"k reset to {self.k =}")

    def load_all_stores(self) -> None:
        """Load all vector stores for all domains."""
        for domain_name in self.domains:
            self.load(domain_name)

    @property
    def domains(self) -> list[str]:
        """Get a list of all configured domains."""
        return list(self.vs_config.domains.keys())

    def load(self, domain_name: str) -> None:
        """Load vector stores for a domain (lazy loading).

        :param domain_name: Name of the domain to load stores for
        """
        assert self.embeddings

        domain = self.vs_config.domains.get(domain_name, None)
        assert domain

        self.logger.debug(f"Got domain information: {domain}")

        stores = domain.vector_stores

        for store in stores:
            if store.loaded_object is not None:
                self.logger.debug(f"Store '{store.name}' already loaded, skipping")
                continue

            store_name = store.name
            self.logger.debug(
                f"Got store for domain {domain_name}: {store_name} ({store.path})"
            )

            store.loaded_object = self._instantiate_store(store.path, store_name)

            self.logger.debug(
                f"Finished loading vector store '{store_name}' from {store.path}"
            )

    def _instantiate_store(self, path: str, name: str):
        """Instantiate a vector store based on the URI scheme in path.

        Expected format: ``"scheme:location"``.

        :param path: URI-style string with scheme prefix
            (e.g. ``"chroma:/path/to/dir"``, ``"qdrant:http://localhost:6333"``,
            ``"pgvector:postgresql://localhost/db"``)
        :param name: Collection name for the vector store
        :returns: Instantiated LangChain VectorStore
        :raises ValueError: If the scheme is missing or unknown
        """
        assert self.embeddings
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
                    self.logger.debug(
                        f"Store path made absolute relative to cwd: {store_dir}"
                    )

                if not store_dir.is_dir():
                    self.logger.error(f"Could not find folder: {store_dir}")
                    raise FileNotFoundError(f"Could not find folder: {store_dir}")

                store_db = store_dir / "chroma.sqlite3"
                if not store_db.is_file():
                    raise FileNotFoundError(f"ChromaDB not found at {store_db}")

                self.logger.debug(
                    f"Loading Chroma vector store '{name}' from {store_dir.absolute()}"
                )

                settings = chromadb.config.Settings(
                    is_persistent=True,
                    persist_directory=str(store_dir.absolute()),
                    anonymized_telemetry=False,
                )
                return Chroma(
                    collection_name=name,
                    embedding_function=self.embeddings,
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
                self.logger.debug(f"Loading Qdrant vector store '{name}' at {location}")
                return QdrantVectorStore(
                    client=client,
                    collection_name=name,
                    embedding=self.embeddings,
                )

            case "pgvector":
                try:
                    from langchain_postgres import PGVector
                except ImportError:
                    raise ImportError(
                        "PGVector backend not installed. "
                        "Install: pip install klea_utils[pgvector]"
                    ) from None

                self.logger.debug(
                    f"Loading PGVector vector store '{name}' with connection {location}"
                )
                return PGVector(
                    collection_name=name,
                    embeddings=self.embeddings,
                    connection=location,
                )

            case _:
                raise ValueError(
                    f"Unknown vector store scheme '{scheme}'. "
                    f"Supported: chroma, qdrant, pgvector"
                )

    def retrieve(self, domain_name: str, query: str) -> list[tuple[Document, float]]:
        """Retrieve documents from vector stores for a query.

        :param domain_name: Name of the domain to search in
        :param query: User query string
        :returns: List of (document, relevance_score) tuples
        """
        self.load(domain_name)

        domain = self.vs_config.domains.get(domain_name, None)
        assert domain
        stores = domain.vector_stores

        res = []

        for store in stores:
            assert store.loaded_object
            data = store.loaded_object.similarity_search_with_relevance_scores(
                query, k=self.k, score_threshold=self.sim_thresh
            )
            self.logger.debug(f"{data =}")
            if len(data) == 0:
                self.logger.warning(
                    f"No data retrieved. Check VS is correctly populated and that "
                    f"the collection name is correct ({store.name})"
                )
            res.extend(data)

        return res
